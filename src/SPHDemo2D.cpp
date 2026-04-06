#include "SPHDemo2D.h"
#include "2D/SPH2D.h"
#include "2D/Spawner2D.h"

#include <algorithm>
#include <iostream>

static const char* vs2D = R"(
#version 450 core
layout(location = 0) in vec2 aPos;
uniform float pointSize;
void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    gl_PointSize = pointSize;
}
)";

static const char* fs2D = R"(
#version 450 core
out vec4 FragColor;
void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) discard;
    FragColor = vec4(0.18, 0.52, 1.0, 1.0);
}
)";

SPHDemo2D::SPHDemo2D(GLFWwindow* window)
{
    m_window = window;

    glEnable(GL_PROGRAM_POINT_SIZE);
    glViewport(0, 0, WIDTH, HEIGHT);

    m_program = CompileProgram(vs2D, fs2D);

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    sph::dim2::Spawner spawner;
    spawner.AddRegion(make_float2(0.f, 1.f), make_float2(12.f, 6.f));
    auto spawnData = spawner.GetSpawnData(20.f);

    m_sim = new sph::dim2::Simulation(spawnData, MAX_PARTICLES);
    m_sim->RegisterGLBuffer(m_vbo);
}

SPHDemo2D::~SPHDemo2D()
{
    delete m_sim;
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteProgram(m_program);
}

void SPHDemo2D::Run()
{
    float lastTime     = (float)glfwGetTime();
    float spawnCooldown = 0.0f;

    while (!glfwWindowShouldClose(m_window))
    {
        float now = (float)glfwGetTime();
        float dt  = std::min(now - lastTime, 1.0f / 30.0f);
        lastTime  = now;

        spawnCooldown -= dt;
        if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && spawnCooldown <= 0.0f)
        {
            double mx, my;
            glfwGetCursorPos(m_window, &mx, &my);

            float wx = ((float)mx / WIDTH  * 2.0f - 1.0f) * (m_sim->GetConfig().boundsX * 0.5f);
            float wy = (1.0f - (float)my / HEIGHT * 2.0f) * (m_sim->GetConfig().boundsY * 0.5f);

            static_cast<sph::dim2::Simulation*>(m_sim)->SpawnParticles(50, wx, wy);
            spawnCooldown = 0.1f;
        }

        m_sim->Update(dt);

        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(m_program);
        glUniform1f(glGetUniformLocation(m_program, "pointSize"), 6.0f);
        glBindVertexArray(m_vao);
        glDrawArrays(GL_POINTS, 0, m_sim->ParticleCount());

        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

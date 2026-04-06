#include "SPHDemo3D.h"
#include "3D/SPH3D.h"
#include "3D/Spawner3D.h"

#include <algorithm>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ---- Shaders ----------------------------------------------------------------

static const char* vsParticles = R"(
#version 450 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aVel;

uniform mat4 view;
uniform mat4 projection;
uniform float pointSize;

out float vSpeed;

void main()
{
    vec4 viewPos = view * vec4(aPos, 1.0);
    gl_Position  = projection * viewPos;
    gl_PointSize = pointSize / -viewPos.z;

    vSpeed = length(aVel);
})";

static const char* fsParticles = R"(
#version 450 core
in float vSpeed;

uniform float velocityMax;
uniform sampler2D colourMap;

out vec4 FragColor;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) discard;

    float t = clamp(vSpeed / velocityMax, 0.0, 1.0);
    t = pow(t, 0.5);

    vec3 color = texture(colourMap, vec2(t, 0.5)).rgb;
    FragColor = vec4(color, 1.0);
})";

static const char* vsBox = R"(
#version 450 core
layout(location = 0) in vec3 aPos;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * vec4(aPos, 1.0);
}
)";

static const char* fsBox = R"(
#version 450 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(0.4, 0.4, 0.4, 1.0);
}
)";

static GLuint CreateGradientTexture()
{
    struct Stop { float t; glm::vec3 c; };

    std::vector<Stop> stops = {
        {0.0f, {0.1f, 0.35f, 0.72f}},
        {0.5f, {0.3f, 1.0f, 0.5f}},
        {0.71f, {1.0f, 0.9f, 0.0f}},
        {1.0f, {1.0f, 0.2f, 0.0f}}
    };

    auto sample = [&](float t)
        {
            for (size_t i = 0; i < stops.size() - 1; i++)
            {
                if (t >= stops[i].t && t <= stops[i + 1].t)
                {
                    float lt = (t - stops[i].t) / (stops[i + 1].t - stops[i].t);
                    return glm::mix(stops[i].c, stops[i + 1].c, lt);
                }
            }
            return stops.back().c;
        };

    const int width = 256;
    std::vector<float> data(width * 3);

    for (int i = 0; i < width; i++)
    {
        float t = i / float(width - 1);
        glm::vec3 c = sample(t);

        data[i * 3 + 0] = c.r;
        data[i * 3 + 1] = c.g;
        data[i * 3 + 2] = c.b;
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, 1, 0, GL_RGB, GL_FLOAT, data.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    return tex;
}

struct VBOParticle
{
    float3 pos;
    float3 vel;
};

SPHDemo3D::SPHDemo3D(GLFWwindow* window)
{
    m_window = window;

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glViewport(0, 0, WIDTH, HEIGHT);

    m_program = CompileProgram(vsParticles, fsParticles);
    m_boxProgram = CompileProgram(vsBox, fsBox);

    // Particle VBO
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(VBOParticle), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VBOParticle), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VBOParticle), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    m_gradientTex = CreateGradientTexture();

    sph::dim3::Spawner spawner;
    spawner.AddRegion(make_float3(6.f, 0.f, 0.f), 3.f);
    spawner.AddRegion(make_float3(-6.0f, 0.0f, 0.0f), 3.0f);
    auto spawnData = spawner.GetSpawnData(2000);

    m_sim = new sph::dim3::Simulation(spawnData, MAX_PARTICLES);
    m_sim->RegisterGLBuffer(m_vbo);

    InitBox();

    glfwSetWindowUserPointer(m_window, this);
    glfwSetMouseButtonCallback(m_window, MouseButtonCallback);
    glfwSetCursorPosCallback(m_window, CursorPosCallback);
    glfwSetScrollCallback(m_window, ScrollCallback);
}

SPHDemo3D::~SPHDemo3D()
{
    delete m_sim;
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteProgram(m_program);
    glDeleteVertexArrays(1, &m_boxVao);
    glDeleteBuffers(1, &m_boxVbo);
    glDeleteBuffers(1, &m_boxEbo);
    glDeleteProgram(m_boxProgram);
}

void SPHDemo3D::Run()
{
    float lastTime = (float)glfwGetTime();

    while (!glfwWindowShouldClose(m_window))
    {
        float now = (float)glfwGetTime();
        float dt  = std::min(now - lastTime, 1.0f / 30.0f);
        lastTime  = now;

        m_sim->Update(dt);
        UpdateCamera();

        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view       = glm::lookAt(m_cameraPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)WIDTH / HEIGHT, 0.1f, 500.0f);

        // Draw particles
        glUseProgram(m_program);
        glUniformMatrix4fv(glGetUniformLocation(m_program, "view"),       1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(m_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // pointSize is a screen-space scale factor: gl_PointSize = pointSize / -viewPos.z
        // A value of ~150 gives ~5px dots at camera distance 30

        glUniform1f(glGetUniformLocation(m_program, "pointSize"), 80.0f);

        glUniform1f(glGetUniformLocation(m_program, "velocityMax"), 12.0f);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_gradientTex);
        glUniform1i(glGetUniformLocation(m_program, "colourMap"), 0);

        glBindVertexArray(m_vao);
        glDrawArrays(GL_POINTS, 0, m_sim->ParticleCount());

        // Draw bounding box
        DrawBox(view, projection);

        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

void SPHDemo3D::InitBox()
{
    const sph::Config& cfg = m_sim->GetConfig();
    float hx = cfg.boundsX * 0.5f;
    float hy = cfg.boundsY * 0.5f;
    float hz = cfg.boundsZ * 0.5f;

    // 8 corners of the bounding box
    float verts[8 * 3] = {
        -hx, -hy, -hz,   // 0
         hx, -hy, -hz,   // 1
         hx,  hy, -hz,   // 2
        -hx,  hy, -hz,   // 3
        -hx, -hy,  hz,   // 4
         hx, -hy,  hz,   // 5
         hx,  hy,  hz,   // 6
        -hx,  hy,  hz,   // 7
    };

    // 12 edges as line pairs
    unsigned int indices[24] = {
        0,1, 1,2, 2,3, 3,0,   // front face
        4,5, 5,6, 6,7, 7,4,   // back face
        0,4, 1,5, 2,6, 3,7    // connecting edges
    };

    glGenVertexArrays(1, &m_boxVao);
    glGenBuffers(1, &m_boxVbo);
    glGenBuffers(1, &m_boxEbo);

    glBindVertexArray(m_boxVao);

    glBindBuffer(GL_ARRAY_BUFFER, m_boxVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_boxEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
}

void SPHDemo3D::DrawBox(const glm::mat4& view, const glm::mat4& proj)
{
    glUseProgram(m_boxProgram);
    glUniformMatrix4fv(glGetUniformLocation(m_boxProgram, "view"),       1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_boxProgram, "projection"), 1, GL_FALSE, glm::value_ptr(proj));

    glBindVertexArray(m_boxVao);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void SPHDemo3D::UpdateCamera()
{
    float yawRad   = glm::radians(m_cameraYaw);
    float pitchRad = glm::radians(m_cameraPitch);
    m_cameraPos.x  = m_cameraDistance * cosf(pitchRad) * cosf(yawRad);
    m_cameraPos.y  = m_cameraDistance * sinf(pitchRad);
    m_cameraPos.z  = m_cameraDistance * cosf(pitchRad) * sinf(yawRad);
}

void SPHDemo3D::MouseButtonCallback(GLFWwindow* w, int button, int action, int mods)
{
    auto* demo = static_cast<SPHDemo3D*>(glfwGetWindowUserPointer(w));
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        demo->m_mousePressed = (action == GLFW_PRESS);
        if (demo->m_mousePressed)
        {
            glfwGetCursorPos(w, &demo->m_lastMouseX, &demo->m_lastMouseY);
            demo->m_firstMouse = true;
        }
    }
}

void SPHDemo3D::CursorPosCallback(GLFWwindow* w, double xpos, double ypos)
{
    auto* demo = static_cast<SPHDemo3D*>(glfwGetWindowUserPointer(w));
    if (!demo->m_mousePressed) return;

    if (demo->m_firstMouse)
    {
        demo->m_lastMouseX = xpos;
        demo->m_lastMouseY = ypos;
        demo->m_firstMouse = false;
        return;
    }

    float xoffset = (float)(xpos - demo->m_lastMouseX);
    float yoffset = (float)(ypos - demo->m_lastMouseY);
    demo->m_lastMouseX = xpos;
    demo->m_lastMouseY = ypos;

    constexpr float sensitivity = 0.4f;
    demo->m_cameraYaw   += xoffset * sensitivity;
    demo->m_cameraPitch += yoffset * sensitivity;
    demo->m_cameraPitch  = std::clamp(demo->m_cameraPitch, -89.0f, 89.0f);
}

void SPHDemo3D::ScrollCallback(GLFWwindow* w, double xoffset, double yoffset)
{
    auto* demo = static_cast<SPHDemo3D*>(glfwGetWindowUserPointer(w));
    demo->m_cameraDistance -= (float)yoffset * 1.5f;
    demo->m_cameraDistance  = std::clamp(demo->m_cameraDistance, 2.0f, 150.0f);
}

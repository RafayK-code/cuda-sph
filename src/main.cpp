#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <algorithm>
#include "SPH.h"

const char* vertexShaderSource = R"(
#version 450 core
layout(location = 0) in vec2 aPos;
uniform float pointSize;
void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    gl_PointSize = pointSize;
}
)";

const char* fragmentShaderSource = R"(
#version 450 core
out vec4 FragColor;
void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    if(length(coord) > 0.5) discard;

    vec3 base = vec3(0.18, 0.52, 1.0);
    FragColor = vec4(base, 1.0);
}
)";

static void compileShader(unsigned int shader, const char* name)
{
    glCompileShader(shader);
    int ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << name << " shader error: " << log << std::endl;
    }
}

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    const int WIDTH = 1280;
    const int HEIGHT = 720;

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "SPH Fluid Sim", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;

    glViewport(0, 0, WIDTH, HEIGHT);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Compile shaders
    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSource, nullptr);
    compileShader(vs, "Vertex");

    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSource, nullptr);
    compileShader(fs, "Fragment");

    unsigned int program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    {
        int ok;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            char log[512];
            glGetProgramInfoLog(program, 512, nullptr, log);
            std::cerr << "Shader link error: " << log << std::endl;
        }
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    // VAO / VBO
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * sizeof(float) * 2, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Init SPH
    sph::Init(2000);
    sph::RegisterGLBuffer(vbo);

    float lastTime = (float)glfwGetTime();
    float spawnCooldown = 0.0f;

    while (!glfwWindowShouldClose(window))
    {
        float now = (float)glfwGetTime();
        float dt = now - lastTime;
        lastTime = now;
        dt = std::min(dt, 1.0f / 30.0f);

        // Mouse spawn
        spawnCooldown -= dt;
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && spawnCooldown <= 0.0f)
        {
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);

            // screen to world space (centered bounds)
            float wx = ((float)mx / WIDTH * 2.0f - 1.0f) * (17.0f * 0.5f);
            float wy = (1.0f - (float)my / HEIGHT * 2.0f) * (9.0f * 0.5f);

            sph::SpawnParticles(50, wx, wy);
            spawnCooldown = 0.1f;
        }

        sph::Update(dt);

        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glUniform1f(glGetUniformLocation(program, "pointSize"), 6.0f);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, sph::ParticleCount());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    sph::Cleanup();
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    glfwTerminate();
    return 0;
}
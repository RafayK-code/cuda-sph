#include "SPHDemo.h"
#include <iostream>

static GLuint CompileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "shader error: " << log << std::endl;
    }

    return shader;
}

GLuint ISPHDemo::CompileProgram(const char* vsSrc, const char* fsSrc)
{
    GLuint vs = CompileShader(GL_VERTEX_SHADER,   vsSrc);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fsSrc);

    GLuint program = glCreateProgram();
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
    return program;
}

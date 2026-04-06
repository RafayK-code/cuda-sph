#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class ISPHDemo
{
public:
    virtual ~ISPHDemo() = default;

    virtual void Run() = 0;

protected:
    static GLuint CompileProgram(const char* vsSrc, const char* fsSrc);

    static constexpr int WIDTH  = 1280;
    static constexpr int HEIGHT = 720;
};

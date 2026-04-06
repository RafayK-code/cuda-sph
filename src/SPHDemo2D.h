#pragma once
#include "SPHDemo.h"
#include "SPH.h"

class SPHDemo2D : public ISPHDemo
{
public:
    SPHDemo2D(GLFWwindow* window);
    ~SPHDemo2D() override;

    void Run() override;

private:
    GLFWwindow* m_window  = nullptr;

    GLuint m_program = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;

    sph::ISimulation* m_sim = nullptr;

    static constexpr int MAX_PARTICLES = 10000;
};

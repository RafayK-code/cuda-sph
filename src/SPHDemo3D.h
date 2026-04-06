#pragma once
#include "SPHDemo.h"
#include "SPH.h"

#include <glm/glm.hpp>

class SPHDemo3D : public ISPHDemo
{
public:
    SPHDemo3D(GLFWwindow* window);
    ~SPHDemo3D() override;

    void Run() override;

private:
    void InitBox();
    void UpdateBox();
    void DrawBox(const glm::mat4& view, const glm::mat4& proj);
    void UpdateCamera();

    static void MouseButtonCallback(GLFWwindow* w, int button, int action, int mods);
    static void CursorPosCallback(GLFWwindow* w, double xpos, double ypos);
    static void ScrollCallback(GLFWwindow* w, double xoffset, double yoffset);

private:
    GLFWwindow* m_window  = nullptr;

    // particle rendering
    GLuint m_program = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;

    // bounds wireframe
    GLuint m_boxProgram = 0;
    GLuint m_boxVao = 0;
    GLuint m_boxVbo = 0;
    GLuint m_boxEbo = 0;

    GLint m_gradientTex;

    sph::ISimulation* m_sim = nullptr;

    float m_cameraYaw = 40.0f;
    float m_cameraPitch = 20.0f;
    float m_cameraDistance = 15.0f;
    glm::vec3 m_cameraPos = glm::vec3(15.0f, 0.0f, 0.0f);

    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    bool m_mousePressed = false;
    bool m_firstMouse = true;

    float m_simBoundsX;
    float m_simBoundsY;
    float m_simBoundsZ;

    static constexpr int MAX_PARTICLES = 100000;
};

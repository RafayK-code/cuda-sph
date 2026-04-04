#pragma once
#include <glad/glad.h>

namespace sph
{
    void Init(int count);
    void RegisterGLBuffer(GLuint vbo);
    void SpawnParticles(int count, float x, float y);
    void Update(float dt);
    void Cleanup();
    int  ParticleCount();
}

constexpr int   MAX_PARTICLES = 10000;
constexpr float SPAWN_SPACING = 0.25f; // world space spacing between particles on spawn

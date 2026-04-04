#pragma once
#include <glad/glad.h>

namespace sph
{
    enum class Dimension
    {
        Dim2, Dim3
    };

    struct Config
    {
        float gravity = -9.8f;
        float collisionDamping = 0.95f;
        float smoothingRadius = 0.35f;
        float targetDensity = 55.0f;
        float pressureMultiplier = 500.0f;
        float nearPressureMultiplier = 5.0f;
        float viscosity = 0.03f;
        float boundsX = 17.0f;
        float boundsY = 9.0f;
        float boundsZ = 0.0f;
        int substeps = 3;
        float fixedDt = 1.0f / 240.0f;
        float spawnSpacing = 0.25f;

        static Config Create(Dimension dim);
    };

    class ISimulation
    {
    public:
        virtual ~ISimulation() = default;

        virtual void Update(float dt) = 0;

        virtual void UpdateConfig(const Config& config) = 0;

        virtual int ParticleCount() const = 0;
        virtual const Config& GetConfig() const = 0;

        virtual void RegisterGLBuffer(GLuint buffer) = 0;
    };
}

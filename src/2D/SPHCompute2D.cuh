#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace sph::dim2
{
    struct Particle
    {
        float2 position;
        float2 predictedPosition;
        float2 velocity;
        float2 acceleration;
        float density;
        float nearDensity;
        float pressure;
        float nearPressure;
        float mass;
    };

    struct DevicePhysicsConfig
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
    };

    void UploadScalingFactors(float radius);
    void UploadPhysicsConfig(const DevicePhysicsConfig& config);

    __global__ void ExternalForcesKernel(Particle* particles, int count, float dt);

    __global__ void ComputeKeysKernel(
        Particle* particles,
        uint32_t* spatialKeys,
        uint32_t* sortBuffer,
        uint32_t* sortedIndices,
        int count
    );

    __global__ void MarkOffsetsKernel(uint32_t* sortBuffer, uint32_t* offsets, int count);

    __global__ void CalculateDensitiesKernel(
        Particle* particles,
        uint32_t* spatialKeys,
        uint32_t* spatialOffsets,
        uint32_t* sortedIndices,
        int count
    );

    __global__ void CalculatePressureKernel(
        Particle* particles,
        uint32_t* spatialKeys,
        uint32_t* spatialOffsets,
        uint32_t* sortedIndices,
        int count, float dt
    );

    __global__ void CalculateViscosityKernel(
        Particle* particles,
        uint32_t* spatialKeys,
        uint32_t* spatialOffsets,
        uint32_t* sortedIndices,
        int count, float dt
    );

    __global__ void UpdatePositionsKernel(Particle* particles, int count, float dt);

    __global__ void CopyToVBOKernel(Particle* particles, float2* vbo, int count);
}
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace sph::dim3
{
    struct Particle
    {
        float3 position;
        float3 predictedPosition;
        float3 velocity;
        float3 acceleration;
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

        bool operator==(const DevicePhysicsConfig& other) const
        {
            return gravity == other.gravity &&
                collisionDamping == other.collisionDamping &&
                smoothingRadius == other.smoothingRadius &&
                targetDensity == other.targetDensity &&
                pressureMultiplier == other.pressureMultiplier &&
                nearPressureMultiplier == other.nearPressureMultiplier &&
                viscosity == other.viscosity;
        }

        bool operator!=(const DevicePhysicsConfig& other) const
        {
            return !(*this == other);
        }
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

    __global__ void UpdatePositionsKernel(Particle* particles, float3 bounds, int count, float dt);

    struct VBOParticle
    {
        float3 pos;
        float3 vel;
    };

    __global__ void CopyToVBOKernel(Particle* particles, VBOParticle* vbo, int count);
}

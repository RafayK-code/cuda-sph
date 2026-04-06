#include "SPHCompute3D.cuh"
#include "SpatialHash3D.cuh"
#include "../FluidMaths.cuh"
#include "../Constants.h"

#include <cmath>

namespace sph::dim3
{
    void UploadScalingFactors(float radius)
    {
        float poly6  = 315.0f / (64.0f * PI * powf(radius, 9.0f));
        float spiky3 = 15.0f  / (PI * powf(radius, 6.0f));
        float spiky2 = 15.0f  / (2.0f * PI * powf(radius, 5.0f));
        float dspiky3 = 45.0f / (PI * powf(radius, 6.0f));
        float dspiky2 = 15.0f / (PI * powf(radius, 5.0f));

        cudaMemcpyToSymbol(d_Poly6ScalingFactor,               &poly6,   sizeof(float));
        cudaMemcpyToSymbol(d_SpikyPow3ScalingFactor,           &spiky3,  sizeof(float));
        cudaMemcpyToSymbol(d_SpikyPow2ScalingFactor,           &spiky2,  sizeof(float));
        cudaMemcpyToSymbol(d_SpikyPow3DerivativeScalingFactor, &dspiky3, sizeof(float));
        cudaMemcpyToSymbol(d_SpikyPow2DerivativeScalingFactor, &dspiky2, sizeof(float));
    }

    __constant__ DevicePhysicsConfig d_config;

    void UploadPhysicsConfig(const DevicePhysicsConfig& config)
    {
        cudaMemcpyToSymbol(d_config, &config, sizeof(DevicePhysicsConfig));
    }

    __global__ void ExternalForcesKernel(Particle* particles, int count, float dt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        particles[i].velocity.y += d_config.gravity * dt;

        const float predFactor = 1.0f / 120.0f;
        particles[i].predictedPosition.x = particles[i].position.x + particles[i].velocity.x * predFactor;
        particles[i].predictedPosition.y = particles[i].position.y + particles[i].velocity.y * predFactor;
        particles[i].predictedPosition.z = particles[i].position.z + particles[i].velocity.z * predFactor;
    }

    __global__ void ComputeKeysKernel(
        Particle* particles,
        uint32_t* spatialKeys,
        uint32_t* sortBuffer,
        uint32_t* sortedIndices,
        int count)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        float3 pos = make_float3(particles[i].predictedPosition.x, particles[i].predictedPosition.y, particles[i].predictedPosition.z);
        int3 cell = GetCell(pos, d_config.smoothingRadius);
        uint32_t hash = HashCell(cell);
        uint32_t key  = KeyFromHash(hash, (uint32_t)count);

        spatialKeys[i]    = key;
        sortBuffer[i]     = key;
        sortedIndices[i]  = (uint32_t)i;
    }

    __global__ void MarkOffsetsKernel(uint32_t* sortBuffer, uint32_t* offsets, int count)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        if (i == 0 || sortBuffer[i] != sortBuffer[i - 1])
            offsets[sortBuffer[i]] = (uint32_t)i;
    }

    __global__ void CalculateDensitiesKernel(
        Particle* particles,
        uint32_t* spatialKeys,
        uint32_t* spatialOffsets,
        uint32_t* sortedIndices,
        int count)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        float3 pos = make_float3(particles[i].predictedPosition.x, particles[i].predictedPosition.y, particles[i].predictedPosition.z);
        int3 originCell = GetCell(pos, d_config.smoothingRadius);
        float sqrRadius = d_config.smoothingRadius * d_config.smoothingRadius;
        float density = 0.0f;
        float nearDensity = 0.0f;

        for (int n = 0; n < 27; n++)
        {
            int3 cell = make_int3(originCell.x + d_offsets3D[n].x, originCell.y + d_offsets3D[n].y, originCell.z + d_offsets3D[n].z);
            uint32_t hash = HashCell(cell);
            uint32_t key  = KeyFromHash(hash, (uint32_t)count);

            uint32_t currIndex = spatialOffsets[key];

            while (currIndex < (uint32_t)count)
            {
                uint32_t neighbourIndex = sortedIndices[currIndex];
                currIndex++;

                uint32_t neighbourKey = spatialKeys[neighbourIndex];
                if (neighbourKey != key) break;

                float3 nPos = make_float3(particles[neighbourIndex].predictedPosition.x, particles[neighbourIndex].predictedPosition.y, particles[neighbourIndex].predictedPosition.z);
                float3 off  = make_float3(nPos.x - pos.x, nPos.y - pos.y, nPos.z - pos.z);
                float sqrDst = off.x * off.x + off.y * off.y + off.z * off.z;
                if (sqrDst > sqrRadius) continue;

                float dst = sqrtf(sqrDst);
                density    += SpikyKernelPow2(dst, d_config.smoothingRadius);
                nearDensity += SpikyKernelPow3(dst, d_config.smoothingRadius);
            }
        }

        particles[i].density    = density;
        particles[i].nearDensity = nearDensity;
    }

    __global__ void CalculatePressureKernel(
        Particle* particles,
        uint32_t* spatialKeys,
        uint32_t* spatialOffsets,
        uint32_t* sortedIndices,
        int count, float dt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        float density    = particles[i].density;
        float nearDensity = particles[i].nearDensity;
        float pressure    = (density - d_config.targetDensity) * d_config.pressureMultiplier;
        float nearPressure = nearDensity * d_config.nearPressureMultiplier;

        float3 pos = make_float3(particles[i].predictedPosition.x, particles[i].predictedPosition.y, particles[i].predictedPosition.z);
        int3 originCell = GetCell(pos, d_config.smoothingRadius);
        float sqrRadius = d_config.smoothingRadius * d_config.smoothingRadius;
        float3 pressureForce = make_float3(0.0f, 0.0f, 0.0f);
        int neighbourCount = 0;

        for (int n = 0; n < 27; n++)
        {
            int3 cell = make_int3(originCell.x + d_offsets3D[n].x, originCell.y + d_offsets3D[n].y, originCell.z + d_offsets3D[n].z);
            uint32_t hash = HashCell(cell);
            uint32_t key  = KeyFromHash(hash, (uint32_t)count);

            uint32_t currIndex = spatialOffsets[key];

            while (currIndex < (uint32_t)count)
            {
                uint32_t neighbourIndex = sortedIndices[currIndex];
                currIndex++;

                if (neighbourIndex == (uint32_t)i) continue;

                uint32_t neighbourKey = spatialKeys[neighbourIndex];
                if (neighbourKey != key) break;

                float3 nPos = make_float3(particles[neighbourIndex].predictedPosition.x, particles[neighbourIndex].predictedPosition.y, particles[neighbourIndex].predictedPosition.z);
                float3 off  = make_float3(nPos.x - pos.x, nPos.y - pos.y, nPos.z - pos.z);
                float sqrDst = off.x * off.x + off.y * off.y + off.z * off.z;
                if (sqrDst > sqrRadius) continue;

                float dst = sqrtf(sqrDst);
                float3 dir = dst > 0.0f ? make_float3(off.x / dst, off.y / dst, off.z / dst) : make_float3(0.0f, 1.0f, 0.0f);

                float nDensity     = particles[neighbourIndex].density;
                float nNearDensity = particles[neighbourIndex].nearDensity;
                float nPressure    = (nDensity - d_config.targetDensity) * d_config.pressureMultiplier;
                float nNearPressure = nNearDensity * d_config.nearPressureMultiplier;

                float sharedP  = (pressure + nPressure) * 0.5f;
                float sharedNP = (nearPressure + nNearPressure) * 0.5f;

                float gradP  = DerivativeSpikyPow2(dst, d_config.smoothingRadius) * sharedP  / fmaxf(nDensity,     0.001f);
                float gradNP = DerivativeSpikyPow3(dst, d_config.smoothingRadius) * sharedNP / fmaxf(nNearDensity, 0.001f);

                pressureForce.x += dir.x * (gradP + gradNP);
                pressureForce.y += dir.y * (gradP + gradNP);
                pressureForce.z += dir.z * (gradP + gradNP);
                neighbourCount++;
            }
        }

        float invDensity = 1.0f / fmaxf(density, 0.001f);
        particles[i].velocity.x += pressureForce.x * invDensity * dt;
        particles[i].velocity.y += pressureForce.y * invDensity * dt;
        particles[i].velocity.z += pressureForce.z * invDensity * dt;

        // Airborne drag: damp spray/isolated particles to prevent them flying off
        if (neighbourCount < 8)
        {
            float drag = 1.0f - 0.75f * dt;
            particles[i].velocity.x *= drag;
            particles[i].velocity.y *= drag;
            particles[i].velocity.z *= drag;
        }
    }

    __global__ void CalculateViscosityKernel(
        Particle* particles,
        uint32_t* spatialKeys,
        uint32_t* spatialOffsets,
        uint32_t* sortedIndices,
        int count, float dt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        float3 pos = make_float3(particles[i].predictedPosition.x, particles[i].predictedPosition.y, particles[i].predictedPosition.z);
        float3 vel = make_float3(particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
        int3 originCell = GetCell(pos, d_config.smoothingRadius);
        float sqrRadius = d_config.smoothingRadius * d_config.smoothingRadius;
        float3 viscosityForce = make_float3(0.0f, 0.0f, 0.0f);

        for (int n = 0; n < 27; n++)
        {
            int3 cell = make_int3(originCell.x + d_offsets3D[n].x, originCell.y + d_offsets3D[n].y, originCell.z + d_offsets3D[n].z);
            uint32_t hash = HashCell(cell);
            uint32_t key  = KeyFromHash(hash, (uint32_t)count);

            uint32_t currIndex = spatialOffsets[key];

            while (currIndex < (uint32_t)count)
            {
                uint32_t neighbourIndex = sortedIndices[currIndex];
                currIndex++;

                if (neighbourIndex == (uint32_t)i) continue;

                uint32_t neighbourKey = spatialKeys[neighbourIndex];
                if (neighbourKey != key) break;

                float3 nPos = make_float3(particles[neighbourIndex].predictedPosition.x, particles[neighbourIndex].predictedPosition.y, particles[neighbourIndex].predictedPosition.z);
                float3 off  = make_float3(nPos.x - pos.x, nPos.y - pos.y, nPos.z - pos.z);
                float sqrDst = off.x * off.x + off.y * off.y + off.z * off.z;
                if (sqrDst > sqrRadius) continue;

                float dst = sqrtf(sqrDst);
                float influence = SmoothingKernelPoly6(dst, d_config.smoothingRadius);
                float3 nVel = make_float3(particles[neighbourIndex].velocity.x, particles[neighbourIndex].velocity.y, particles[neighbourIndex].velocity.z);
                viscosityForce.x += (nVel.x - vel.x) * influence;
                viscosityForce.y += (nVel.y - vel.y) * influence;
                viscosityForce.z += (nVel.z - vel.z) * influence;
            }
        }

        particles[i].velocity.x += viscosityForce.x * d_config.viscosity * dt;
        particles[i].velocity.y += viscosityForce.y * d_config.viscosity * dt;
        particles[i].velocity.z += viscosityForce.z * d_config.viscosity * dt;
    }

    __global__ void UpdatePositionsKernel(Particle* particles, float3 bounds, int count, float dt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        particles[i].position.x += particles[i].velocity.x * dt;
        particles[i].position.y += particles[i].velocity.y * dt;
        particles[i].position.z += particles[i].velocity.z * dt;

        float halfX = bounds.x * 0.5f;
        float halfY = bounds.y * 0.5f;
        float halfZ = bounds.z * 0.5f;

        if (particles[i].position.x < -halfX) { particles[i].position.x = -halfX; particles[i].velocity.x *= -d_config.collisionDamping; }
        if (particles[i].position.x >  halfX) { particles[i].position.x =  halfX; particles[i].velocity.x *= -d_config.collisionDamping; }
        if (particles[i].position.y < -halfY) { particles[i].position.y = -halfY; particles[i].velocity.y *= -d_config.collisionDamping; }
        if (particles[i].position.y >  halfY) { particles[i].position.y =  halfY; particles[i].velocity.y *= -d_config.collisionDamping; }
        if (particles[i].position.z < -halfZ) { particles[i].position.z = -halfZ; particles[i].velocity.z *= -d_config.collisionDamping; }
        if (particles[i].position.z >  halfZ) { particles[i].position.z =  halfZ; particles[i].velocity.z *= -d_config.collisionDamping; }
    }

    __global__ void CopyToVBOKernel(Particle* particles, VBOParticle* vbo, int count)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        vbo[i].pos = particles[i].position;
        vbo[i].vel = particles[i].velocity;
    }
}

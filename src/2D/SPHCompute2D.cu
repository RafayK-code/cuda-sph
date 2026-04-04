#include "SPHCompute2D.cuh"
#include "SpatialHash2D.cuh"
#include "../FluidMaths.cuh"
#include "../SPHParams.h"

#include <cmath>

namespace sph::dim2
{
    void UploadScalingFactors(float radius)
    {
        float poly6 = 4.0f / (PI * powf(radius, 8.0f));
        float spiky3 = 10.0f / (PI * powf(radius, 5.0f));
        float spiky2 = 6.0f / (PI * powf(radius, 4.0f));
        float dspiky3 = 30.0f / (PI * powf(radius, 5.0f));
        float dspiky2 = 12.0f / (PI * powf(radius, 4.0f));

        cudaMemcpyToSymbol(d_Poly6ScalingFactor, &poly6, sizeof(float));
        cudaMemcpyToSymbol(d_SpikyPow3ScalingFactor, &spiky3, sizeof(float));
        cudaMemcpyToSymbol(d_SpikyPow2ScalingFactor, &spiky2, sizeof(float));
        cudaMemcpyToSymbol(d_SpikyPow3DerivativeScalingFactor, &dspiky3, sizeof(float));
        cudaMemcpyToSymbol(d_SpikyPow2DerivativeScalingFactor, &dspiky2, sizeof(float));
    }

    __global__ void ExternalForcesKernel(Particle* particles, int count, float dt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        particles[i].velocity.y += GRAVITY * dt;

        const float predFactor = 1.0f / 120.0f;
        particles[i].predictedPosition.x = particles[i].position.x + particles[i].velocity.x * predFactor;
        particles[i].predictedPosition.y = particles[i].position.y + particles[i].velocity.y * predFactor;
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

        float2 pos = make_float2(particles[i].predictedPosition.x, particles[i].predictedPosition.y);
        int2 cell = GetCell2D(pos, SMOOTHING_RADIUS);
        uint32_t  hash = HashCell2D(cell);
        uint32_t  key = KeyFromHash(hash, (uint32_t)count);

        spatialKeys[i] = key;
        sortBuffer[i] = key;
        sortedIndices[i] = (uint32_t)i;
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

        float2 pos = make_float2(particles[i].predictedPosition.x, particles[i].predictedPosition.y);
        int2 originCell = GetCell2D(pos, SMOOTHING_RADIUS);
        float sqrRadius = SMOOTHING_RADIUS * SMOOTHING_RADIUS;
        float density = 0.0f;
        float nearDensity = 0.0f;

        for (int n = 0; n < 9; n++)
        {
            int2 cell = make_int2(originCell.x + d_offsets2D[n].x, originCell.y + d_offsets2D[n].y);
            uint32_t hash = HashCell2D(cell);
            uint32_t key = KeyFromHash(hash, (uint32_t)count);

            uint32_t currIndex = spatialOffsets[key];

            while (currIndex < (uint32_t)count)
            {
                uint32_t neighbourIndex = sortedIndices[currIndex];
                currIndex++;

                uint32_t neighbourKey = spatialKeys[neighbourIndex];
                if (neighbourKey != key) break;

                float2 nPos = make_float2(particles[neighbourIndex].predictedPosition.x, particles[neighbourIndex].predictedPosition.y);
                float2 off = make_float2(nPos.x - pos.x, nPos.y - pos.y);
                float sqrDst = off.x * off.x + off.y * off.y;
                if (sqrDst > sqrRadius) continue;

                float dst = sqrtf(sqrDst);
                density += SpikyKernelPow2(dst, SMOOTHING_RADIUS);
                nearDensity += SpikyKernelPow3(dst, SMOOTHING_RADIUS);
            }
        }

        particles[i].density = density;
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

        float density = particles[i].density;
        float nearDensity = particles[i].nearDensity;
        float pressure = (density - TARGET_DENSITY) * PRESSURE_MULT;
        float nearPressure = nearDensity * NEAR_PRESSURE_MULT;

        float2 pos = make_float2(particles[i].predictedPosition.x, particles[i].predictedPosition.y);
        int2 originCell = GetCell2D(pos, SMOOTHING_RADIUS);
        float sqrRadius = SMOOTHING_RADIUS * SMOOTHING_RADIUS;
        float2 pressureForce = make_float2(0.0f, 0.0f);

        for (int n = 0; n < 9; n++)
        {
            int2 cell = make_int2(originCell.x + d_offsets2D[n].x, originCell.y + d_offsets2D[n].y);
            uint32_t hash = HashCell2D(cell);
            uint32_t key = KeyFromHash(hash, (uint32_t)count);

            uint32_t currIndex = spatialOffsets[key];

            while (currIndex < (uint32_t)count)
            {
                uint32_t neighbourIndex = sortedIndices[currIndex];
                currIndex++;

                if (neighbourIndex == (uint32_t)i) continue;

                uint32_t neighbourKey = spatialKeys[neighbourIndex];
                if (neighbourKey != key) break;

                float2 nPos = make_float2(particles[neighbourIndex].predictedPosition.x, particles[neighbourIndex].predictedPosition.y);
                float2 off = make_float2(nPos.x - pos.x, nPos.y - pos.y);
                float sqrDst = off.x * off.x + off.y * off.y;
                if (sqrDst > sqrRadius) continue;

                float dst = sqrtf(sqrDst);
                float2 dir = dst > 0.0f ? make_float2(off.x / dst, off.y / dst) : make_float2(0.0f, 1.0f);

                float nDensity = particles[neighbourIndex].density;
                float nNearDensity = particles[neighbourIndex].nearDensity;
                float nPressure = (nDensity - TARGET_DENSITY) * PRESSURE_MULT;
                float nNearPressure = nNearDensity * NEAR_PRESSURE_MULT;

                float sharedP = (pressure + nPressure) * 0.5f;
                float sharedNP = (nearPressure + nNearPressure) * 0.5f;

                pressureForce.x += dir.x * DerivativeSpikyPow2(dst, SMOOTHING_RADIUS) * sharedP / fmaxf(nDensity, 0.001f);
                pressureForce.y += dir.y * DerivativeSpikyPow2(dst, SMOOTHING_RADIUS) * sharedP / fmaxf(nDensity, 0.001f);
                pressureForce.x += dir.x * DerivativeSpikyPow3(dst, SMOOTHING_RADIUS) * sharedNP / fmaxf(nNearDensity, 0.001f);
                pressureForce.y += dir.y * DerivativeSpikyPow3(dst, SMOOTHING_RADIUS) * sharedNP / fmaxf(nNearDensity, 0.001f);
            }
        }

        float2 acceleration = make_float2(pressureForce.x / fmaxf(density, 0.001f), pressureForce.y / fmaxf(density, 0.001f));
        particles[i].velocity.x += acceleration.x * dt;
        particles[i].velocity.y += acceleration.y * dt;
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

        float2 pos = make_float2(particles[i].predictedPosition.x, particles[i].predictedPosition.y);
        float2 vel = make_float2(particles[i].velocity.x, particles[i].velocity.y);
        int2 originCell = GetCell2D(pos, SMOOTHING_RADIUS);
        float sqrRadius = SMOOTHING_RADIUS * SMOOTHING_RADIUS;
        float2 viscosityForce = make_float2(0.0f, 0.0f);

        for (int n = 0; n < 9; n++)
        {
            int2 cell = make_int2(originCell.x + d_offsets2D[n].x, originCell.y + d_offsets2D[n].y);
            uint32_t hash = HashCell2D(cell);
            uint32_t key = KeyFromHash(hash, (uint32_t)count);

            uint32_t currIndex = spatialOffsets[key];

            while (currIndex < (uint32_t)count)
            {
                uint32_t neighbourIndex = sortedIndices[currIndex];
                currIndex++;

                if (neighbourIndex == (uint32_t)i) continue;

                uint32_t neighbourKey = spatialKeys[neighbourIndex];
                if (neighbourKey != key) break;

                float2 nPos = make_float2(particles[neighbourIndex].predictedPosition.x, particles[neighbourIndex].predictedPosition.y);
                float2 off = make_float2(nPos.x - pos.x, nPos.y - pos.y);
                float sqrDst = off.x * off.x + off.y * off.y;
                if (sqrDst > sqrRadius) continue;

                float dst = sqrtf(sqrDst);
                float2 nVel = make_float2(particles[neighbourIndex].velocity.x, particles[neighbourIndex].velocity.y);
                viscosityForce.x += (nVel.x - vel.x) * SmoothingKernelPoly6(dst, SMOOTHING_RADIUS);
                viscosityForce.y += (nVel.y - vel.y) * SmoothingKernelPoly6(dst, SMOOTHING_RADIUS);
            }
        }

        particles[i].velocity.x += viscosityForce.x * VISCOSITY * dt;
        particles[i].velocity.y += viscosityForce.y * VISCOSITY * dt;
    }

    __global__ void UpdatePositionsKernel(Particle* particles, int count, float dt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        particles[i].position.x += particles[i].velocity.x * dt;
        particles[i].position.y += particles[i].velocity.y * dt;

        float halfX = BOUNDS_X * 0.5f;
        float halfY = BOUNDS_Y * 0.5f;

        if (particles[i].position.x < -halfX) { particles[i].position.x = -halfX; particles[i].velocity.x *= -COLLISION_DAMPING; }
        if (particles[i].position.x > halfX) { particles[i].position.x = halfX; particles[i].velocity.x *= -COLLISION_DAMPING; }
        if (particles[i].position.y < -halfY) { particles[i].position.y = -halfY; particles[i].velocity.y *= -COLLISION_DAMPING; }
        if (particles[i].position.y > halfY) { particles[i].position.y = halfY; particles[i].velocity.y *= -COLLISION_DAMPING; }
    }

    __global__ void CopyToVBOKernel(Particle* particles, float2* vbo, int count)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        vbo[i].x = particles[i].position.x / (BOUNDS_X * 0.5f);
        vbo[i].y = particles[i].position.y / (BOUNDS_Y * 0.5f);
    }
}
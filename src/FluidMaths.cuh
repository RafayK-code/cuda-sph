#pragma once

#include <cuda_runtime.h>

__constant__ float d_Poly6ScalingFactor;
__constant__ float d_SpikyPow3ScalingFactor;
__constant__ float d_SpikyPow2ScalingFactor;
__constant__ float d_SpikyPow3DerivativeScalingFactor;
__constant__ float d_SpikyPow2DerivativeScalingFactor;

__device__ inline float LinearKernel(float dst, float radius)
{
	if (dst < radius)
    {
        return 1.0f - dst / radius;
    }
    return 0.0f;
}

__device__ inline float SmoothingKernelPoly6(float dst, float radius)
{
    if (dst < radius)
    {
        float v = radius * radius - dst * dst;
        return v * v * v * d_Poly6ScalingFactor;
    }
    return 0.0f;
}

__device__ inline float SpikyKernelPow3(float dst, float radius)
{
    if (dst < radius)
    {
        float v = radius - dst;
        return v * v * v * d_SpikyPow3ScalingFactor;
    }
    return 0.0f;
}

__device__ inline float SpikyKernelPow2(float dst, float radius)
{
    if (dst < radius)
    {
        float v = radius - dst;
        return v * v * d_SpikyPow2ScalingFactor;
    }
    return 0.0f;
}

__device__ inline float DerivativeSpikyPow3(float dst, float radius)
{
    if (dst <= radius)
    {
        float v = radius - dst;
        return -v * v * d_SpikyPow3DerivativeScalingFactor;
    }
    return 0.0f;
}

__device__ inline float DerivativeSpikyPow2(float dst, float radius)
{
    if (dst <= radius)
    {
        float v = radius - dst;
        return -v * d_SpikyPow2DerivativeScalingFactor;
    }
    return 0.0f;
}

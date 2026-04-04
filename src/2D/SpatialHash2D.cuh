#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace sph::dim2
{
    __constant__ int2 d_offsets2D[9] = {
        {-1, -1}, {0, -1}, {1, -1},
        {-1,  0}, {0,  0}, {1,  0},
        {-1,  1}, {0,  1}, {1,  1}
    };

    __device__ inline int2 GetCell(float2 pos, float radius)
    {
        return make_int2((int)floorf(pos.x / radius), (int)floorf(pos.y / radius));
    }

    __device__ inline uint32_t HashCell(int2 cell)
    {
        uint32_t a = (uint32_t)cell.x * 1893067u;
        uint32_t b = (uint32_t)cell.y * 4578203u;
        return a + b;
    }

    __device__ inline uint32_t KeyFromHash(uint32_t hash, uint32_t numParticles)
    {
        return hash % numParticles;
    }
}

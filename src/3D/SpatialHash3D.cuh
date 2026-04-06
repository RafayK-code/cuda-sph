#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace sph::dim3
{
    __constant__ int3 d_offsets3D[27] = {
        {-1, -1, -1}, {0, -1, -1}, {1, -1, -1},
        {-1,  0, -1}, {0,  0, -1}, {1,  0, -1},
        {-1,  1, -1}, {0,  1, -1}, {1,  1, -1},

        {-1, -1,  0}, {0, -1,  0}, {1, -1,  0},
        {-1,  0,  0}, {0,  0,  0}, {1,  0,  0},
        {-1,  1,  0}, {0,  1,  0}, {1,  1,  0},

        {-1, -1,  1}, {0, -1,  1}, {1, -1,  1},
        {-1,  0,  1}, {0,  0,  1}, {1,  0,  1},
        {-1,  1,  1}, {0,  1,  1}, {1,  1,  1}
    };

    static constexpr uint32_t hashK1 = 15823u;
    static constexpr uint32_t hashK2 = 9737333u;
    static constexpr uint32_t hashK3 = 440817757u;

    __device__ inline int3 GetCell(float3 pos, float radius)
    {
        return make_int3((int)floorf(pos.x / radius), (int)floorf(pos.y / radius), (int)floorf(pos.z / radius));
    }

    __device__ inline uint32_t HashCell(int3 cell)
    {
        static constexpr uint32_t blockSize = 50u;

        uint3 ucell = make_uint3(
            (uint32_t)(cell.x + (int)(blockSize / 2)),
            (uint32_t)(cell.y + (int)(blockSize / 2)),
            (uint32_t)(cell.z + (int)(blockSize / 2))
        );

        uint3 localCell = make_uint3(ucell.x % blockSize, ucell.y % blockSize, ucell.z % blockSize);
        uint3 blockID   = make_uint3(ucell.x / blockSize, ucell.y / blockSize, ucell.z / blockSize);

        uint32_t blockHash = blockID.x * hashK1 + blockID.y * hashK2 + blockID.z * hashK3;
        return localCell.x + blockSize * (localCell.y + blockSize * localCell.z) + blockHash;
    }

    __device__ inline uint32_t KeyFromHash(uint32_t hash, uint32_t numParticles)
    {
        return hash % numParticles;
    }
}

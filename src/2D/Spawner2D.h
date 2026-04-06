#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace sph::dim2
{
    struct SpawnData
    {
        std::vector<float2> positions;
        std::vector<float2> velocities;
    };

    struct SpawnRegion
    {
        float2 centre;
        float2 size;
    };

    class Spawner
    {
    public:
        void AddRegion(float2 centre, float2 size);

        // spawnDensity: target particles per unit area
        // initialVel:  starting velocity for all particles
        // jitter:      max random offset applied to each particle position
        SpawnData GetSpawnData(float spawnDensity,
                               float2 initialVel = {0.f, 0.f},
                               float  jitter     = 0.f) const;

    private:
        std::vector<SpawnRegion> m_regions;
    };
}

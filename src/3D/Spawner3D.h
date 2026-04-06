#pragma once
#include <cuda_runtime.h>
#include <vector>

namespace sph::dim3
{
    struct SpawnData
    {
        std::vector<float3> positions;
        std::vector<float3> velocities;
    };

    struct SpawnRegion
    {
        float3 centre;
        float  size;
    };

    class Spawner
    {
    public:
        void AddRegion(float3 centre, float size);

        // spawnDensity: target particles per unit volume
        // initialVel:  starting velocity for all particles
        // jitter:      max random offset applied to each particle position
        SpawnData GetSpawnData(int    spawnDensity,
                               float3 initialVel = {0.f, 0.f, 0.f},
                               float  jitter     = 0.f) const;

    private:
        std::vector<SpawnRegion> m_regions;
    };
}

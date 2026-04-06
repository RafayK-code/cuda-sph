#include "Spawner3D.h"

#include <cmath>
#include <random>

namespace sph::dim3
{
    void Spawner::AddRegion(float3 centre, float size)
    {
        m_regions.push_back({centre, size});
    }

    SpawnData Spawner::GetSpawnData(int spawnDensity, float3 initialVel, float jitter) const
    {
        SpawnData result;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> jitterDist(-1.f, 1.f);

        for (const SpawnRegion& region : m_regions)
        {
            float volume    = region.size * region.size * region.size;
            int   target    = (int)(volume * spawnDensity);
            int   perAxis   = (int)cbrtf((float)target);
            if (perAxis < 1) perAxis = 1;

            for (int x = 0; x < perAxis; x++)
            {
                for (int y = 0; y < perAxis; y++)
                {
                    for (int z = 0; z < perAxis; z++)
                    {
                        float u  = perAxis > 1 ? x / (perAxis - 1.f) : 0.5f;
                        float v  = perAxis > 1 ? y / (perAxis - 1.f) : 0.5f;
                        float w  = perAxis > 1 ? z / (perAxis - 1.f) : 0.5f;

                        float px = (u - 0.5f) * region.size + region.centre.x;
                        float py = (v - 0.5f) * region.size + region.centre.y;
                        float pz = (w - 0.5f) * region.size + region.centre.z;

                        if (jitter > 0.f)
                        {
                            px += jitterDist(rng) * jitter;
                            py += jitterDist(rng) * jitter;
                            pz += jitterDist(rng) * jitter;
                        }

                        result.positions.push_back(make_float3(px, py, pz));
                        result.velocities.push_back(initialVel);
                    }
                }
            }
        }

        return result;
    }
}

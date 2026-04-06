#include "Spawner2D.h"
#include "../Constants.h"

#include <cmath>
#include <random>

namespace sph::dim2
{
    void Spawner::AddRegion(float2 centre, float2 size)
    {
        m_regions.push_back({centre, size});
    }

    SpawnData Spawner::GetSpawnData(float spawnDensity, float2 initialVel, float jitter) const
    {
        SpawnData result;

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> angleDist(0.f, 2 * PI);
        std::uniform_real_distribution<float> magDist(-0.5f, 0.5f);

        for (const SpawnRegion& region : m_regions)
        {
            // Sebastian's per-axis count formula: preserve aspect ratio while
            // hitting the target particle count (area * density)
            float area        = region.size.x * region.size.y;
            int   targetTotal = (int)ceilf(area * spawnDensity);
            float lenSum      = region.size.x + region.size.y;
            float tx          = region.size.x / lenSum;
            float ty          = region.size.y / lenSum;
            float m           = sqrtf((float)targetTotal / (tx * ty));
            int   nx          = (int)ceilf(tx * m);
            int   ny          = (int)ceilf(ty * m);

            for (int y = 0; y < ny; y++)
            {
                for (int x = 0; x < nx; x++)
                {
                    float u  = nx > 1 ? x / (nx - 1.f) : 0.5f;
                    float v  = ny > 1 ? y / (ny - 1.f) : 0.5f;
                    float px = (u - 0.5f) * region.size.x + region.centre.x;
                    float py = (v - 0.5f) * region.size.y + region.centre.y;

                    if (jitter > 0.f)
                    {
                        float  angle = angleDist(rng);
                        float2 dir   = make_float2(cosf(angle), sinf(angle));
                        float  mag   = magDist(rng) * jitter;
                        px += dir.x * mag;
                        py += dir.y * mag;
                    }

                    result.positions.push_back(make_float2(px, py));
                    result.velocities.push_back(initialVel);
                }
            }
        }

        return result;
    }
}

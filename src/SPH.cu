#include "SPH.h"

sph::Config sph::Config::Create(Dimension dim)
{
    Config c;

    switch (dim)
    {
    case Dimension::Dim2:
        c.gravity = -9.8f;
        c.collisionDamping = 0.95f;
        c.smoothingRadius = 0.35f;
        c.targetDensity = 55.0f;
        c.pressureMultiplier = 500.0f;
        c.nearPressureMultiplier = 5.0f;
        c.viscosity = 0.03f;
        c.boundsX = 17.0f;
        c.boundsY = 9.0f;
        c.boundsZ = 0.0f;
        c.substeps = 3;
        c.fixedDt = 1.0f / 60.0f;
        c.spawnSpacing = 0.25f;
        break;
    case Dimension::Dim3:
        c.gravity = -9.8f;
        c.collisionDamping = 0.95f;
        c.smoothingRadius = 0.2f;
        c.targetDensity = 630.0f;
        c.pressureMultiplier = 288.0f;
        c.nearPressureMultiplier = 2.16f;
        c.viscosity = 0.0f;
        c.boundsX = 15.0f;
        c.boundsY = 5.0f;
        c.boundsZ = 8.0f;
        c.substeps = 3;
        c.fixedDt = 1.0f / 60.0f;
        c.spawnSpacing = 0.15f;
        break;
    }

    return c;
}
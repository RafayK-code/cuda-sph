#include "SPH.h"
#include "FluidMaths.cuh"
#include "2D/SpatialHash2D.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include <vector>
#include <cmath>

#include "SPHParams.h"
#include "2D/SPHCompute2D.cuh"

static sph::dim2::Particle* d_particles = nullptr;
static uint32_t* d_spatialKeys = nullptr;
static uint32_t* d_sortBuffer = nullptr;
static uint32_t* d_spatialOffsets = nullptr;
static uint32_t* d_sortedIndices = nullptr;

static int g_particleCount = 0;
static cudaGraphicsResource* g_vboResource = nullptr;

void sph::Init(int count)
{
    dim2::UploadScalingFactors(SMOOTHING_RADIUS);

    cudaMalloc((void**)&d_particles, MAX_PARTICLES * sizeof(dim2::Particle));
    cudaMalloc((void**)&d_spatialKeys, MAX_PARTICLES * sizeof(uint32_t));
    cudaMalloc((void**)&d_sortBuffer, MAX_PARTICLES * sizeof(uint32_t));
    cudaMalloc((void**)&d_spatialOffsets, MAX_PARTICLES * sizeof(uint32_t));
    cudaMalloc((void**)&d_sortedIndices, MAX_PARTICLES * sizeof(uint32_t));

    std::vector<dim2::Particle> h(MAX_PARTICLES);

    int cols = (int)sqrtf((float)count);
    float startX = -(cols * SPAWN_SPACING) * 0.5f;
    float startY = (count / cols * SPAWN_SPACING) * 0.5f;

    for (int idx = 0; idx < count; idx++)
    {
        int x = idx % cols;
        int y = idx / cols;
        h[idx].position = make_float2(startX + x * SPAWN_SPACING, startY - y * SPAWN_SPACING);
        h[idx].predictedPosition = h[idx].position;
        h[idx].velocity = make_float2(0.0f, 0.0f);
        h[idx].acceleration = make_float2(0.0f, 0.0f);
        h[idx].density = TARGET_DENSITY;
        h[idx].nearDensity = 0.0f;
        h[idx].pressure = 0.0f;
        h[idx].nearPressure = 0.0f;
        h[idx].mass = 1.0f;
    }

    g_particleCount = count;
    cudaMemcpy(d_particles, h.data(), MAX_PARTICLES * sizeof(dim2::Particle), cudaMemcpyHostToDevice);
}

void sph::RegisterGLBuffer(GLuint vbo)
{
    cudaGraphicsGLRegisterBuffer(&g_vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

#undef min

void sph::SpawnParticles(int count, float wx, float wy)
{
    int toSpawn = std::min(count, MAX_PARTICLES - g_particleCount);
    if (toSpawn <= 0) return;

    std::vector<dim2::Particle> h(toSpawn);
    int cols = (int)sqrtf((float)toSpawn);

    for (int i = 0; i < toSpawn; i++)
    {
        int x = i % cols;
        int y = i / cols;
        h[i].position = make_float2(wx + x * SPAWN_SPACING, wy + y * SPAWN_SPACING);
        h[i].predictedPosition = h[i].position;
        h[i].velocity = make_float2(0.0f, 0.0f);
        h[i].acceleration = make_float2(0.0f, 0.0f);
        h[i].density = TARGET_DENSITY;
        h[i].nearDensity = 0.0f;
        h[i].pressure = 0.0f;
        h[i].nearPressure = 0.0f;
        h[i].mass = 1.0f;
    }

    cudaMemcpy(d_particles + g_particleCount, h.data(), toSpawn * sizeof(dim2::Particle), cudaMemcpyHostToDevice);
    g_particleCount += toSpawn;
}

void sph::Update(float dt)
{
    int threads = 256;
    int blocks = (g_particleCount + threads - 1) / threads;

    constexpr int   SUBSTEPS = 3;
    constexpr float STEP_DT = FIXED_DT / SUBSTEPS;

    for (int step = 0; step < SUBSTEPS; step++)
    {
        dim2::ExternalForcesKernel<<<blocks, threads>>>(d_particles, g_particleCount, STEP_DT);
        cudaDeviceSynchronize();

        dim2::ComputeKeysKernel<<<blocks, threads>>>(d_particles, d_spatialKeys, d_sortBuffer, d_sortedIndices, g_particleCount);
        cudaDeviceSynchronize();

        thrust::device_ptr<uint32_t> sort_ptr(d_sortBuffer);
        thrust::device_ptr<uint32_t> idx_ptr(d_sortedIndices);
        thrust::sort_by_key(sort_ptr, sort_ptr + g_particleCount, idx_ptr);

        cudaMemset(d_spatialOffsets, 0xFF, g_particleCount * sizeof(uint32_t));
        dim2::MarkOffsetsKernel<<<blocks, threads>>>(d_sortBuffer, d_spatialOffsets, g_particleCount);
        cudaDeviceSynchronize();

        dim2::CalculateDensitiesKernel<<<blocks, threads>>>(
            d_particles, d_spatialKeys, d_spatialOffsets, d_sortedIndices, g_particleCount);
        cudaDeviceSynchronize();

        dim2::CalculatePressureKernel<<<blocks, threads>>>(
            d_particles, d_spatialKeys, d_spatialOffsets, d_sortedIndices, g_particleCount, STEP_DT);
        cudaDeviceSynchronize();

        dim2::CalculateViscosityKernel<<<blocks, threads>>>(
            d_particles, d_spatialKeys, d_spatialOffsets, d_sortedIndices, g_particleCount, STEP_DT);
        cudaDeviceSynchronize();

        dim2::UpdatePositionsKernel<<<blocks, threads>>>(d_particles, g_particleCount, STEP_DT);
        cudaDeviceSynchronize();
    }

    // copy to VBO
    float2* d_vbo = nullptr;
    size_t size = 0;
    cudaGraphicsMapResources(1, &g_vboResource);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &size, g_vboResource);
    dim2::CopyToVBOKernel<<<blocks, threads>>>(d_particles, d_vbo, g_particleCount);
    cudaGraphicsUnmapResources(1, &g_vboResource);
}

int sph::ParticleCount()
{
    return g_particleCount;
}

void sph::Cleanup()
{
    if (d_particles) { cudaFree(d_particles);      d_particles = nullptr; }
    if (d_spatialKeys) { cudaFree(d_spatialKeys);    d_spatialKeys = nullptr; }
    if (d_sortBuffer) { cudaFree(d_sortBuffer);     d_sortBuffer = nullptr; }
    if (d_spatialOffsets) { cudaFree(d_spatialOffsets); d_spatialOffsets = nullptr; }
    if (d_sortedIndices) { cudaFree(d_sortedIndices);  d_sortedIndices = nullptr; }
    if (g_vboResource) { cudaGraphicsUnregisterResource(g_vboResource); g_vboResource = nullptr; }
}

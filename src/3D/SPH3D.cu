#include "SPH3D.h"

#include "../FluidMaths.cuh"
#include "SpatialHash3D.cuh"
#include "SPHCompute3D.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include <vector>
#include <cmath>

namespace sph::dim3
{
    static DevicePhysicsConfig g_devicePhysicsConfig;

    static DevicePhysicsConfig CreateDevicePhysicsConfig(const Config& config)
    {
        DevicePhysicsConfig dconf;
        dconf.gravity = config.gravity;
        dconf.collisionDamping = config.collisionDamping;
        dconf.smoothingRadius = config.smoothingRadius;
        dconf.targetDensity = config.targetDensity;
        dconf.pressureMultiplier = config.pressureMultiplier;
        dconf.nearPressureMultiplier = config.nearPressureMultiplier;
        dconf.viscosity = config.viscosity;
        dconf.boundsX = config.boundsX;
        dconf.boundsY = config.boundsY;
        dconf.boundsZ = config.boundsZ;

        return dconf;
    }

    struct Simulation::DeviceData
    {
        Particle* d_particles = nullptr;
        uint32_t* d_spatialKeys = nullptr;
        uint32_t* d_sortBuffer = nullptr;
        uint32_t* d_spatialOffsets = nullptr;
        uint32_t* d_sortedIndices = nullptr;
        cudaGraphicsResource* d_vboResource = nullptr;

        DeviceData(size_t maxParticles)
        {
            cudaMalloc((void**)&d_particles, maxParticles * sizeof(Particle));
            cudaMalloc((void**)&d_spatialKeys, maxParticles * sizeof(uint32_t));
            cudaMalloc((void**)&d_sortBuffer, maxParticles * sizeof(uint32_t));
            cudaMalloc((void**)&d_spatialOffsets, maxParticles * sizeof(uint32_t));
            cudaMalloc((void**)&d_sortedIndices, maxParticles * sizeof(uint32_t));
        }

        ~DeviceData()
        {
            cudaFree(d_particles);
            cudaFree(d_spatialKeys);
            cudaFree(d_sortBuffer);
            cudaFree(d_spatialOffsets);
            cudaFree(d_sortedIndices);
        }
    };

    Simulation::Simulation(int initialParticles, int maxParticles)
        : m_config(Config::Create(Dimension::Dim3))
        , m_maxParticles(maxParticles)
        , m_particleCount(initialParticles)
    {
        UploadScalingFactors(m_config.smoothingRadius);
        g_devicePhysicsConfig = CreateDevicePhysicsConfig(m_config);

        UploadPhysicsConfig(g_devicePhysicsConfig);

        m_data = new DeviceData(m_maxParticles);
        InitSpawnParticles();
    }

    Simulation::Simulation(const SpawnData& spawnData, int maxParticles)
        : m_config(Config::Create(Dimension::Dim3))
        , m_maxParticles(maxParticles)
        , m_particleCount(std::min((int)spawnData.positions.size(), maxParticles))
    {
        UploadScalingFactors(m_config.smoothingRadius);
        g_devicePhysicsConfig = CreateDevicePhysicsConfig(m_config);
        UploadPhysicsConfig(g_devicePhysicsConfig);

        m_data = new DeviceData(m_maxParticles);

        std::vector<dim3::Particle> h(m_maxParticles);
        for (int i = 0; i < m_particleCount; i++)
        {
            h[i].position          = spawnData.positions[i];
            h[i].predictedPosition = spawnData.positions[i];
            h[i].velocity          = spawnData.velocities[i];
            h[i].acceleration      = make_float3(0.f, 0.f, 0.f);
            h[i].density           = m_config.targetDensity;
            h[i].nearDensity       = 0.f;
            h[i].pressure          = 0.f;
            h[i].nearPressure      = 0.f;
            h[i].mass              = 1.f;
        }
        cudaMemcpy(m_data->d_particles, h.data(), m_maxParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    }

    Simulation::~Simulation()
    {
        delete m_data;
    }

    void Simulation::Update(float dt)
    {
        int threads = 256;
        int blocks = (m_particleCount + threads - 1) / threads;

        float stepDt = m_config.fixedDt / m_config.substeps;

        for (int step = 0; step < m_config.substeps; step++)
        {
            ExternalForcesKernel<<<blocks, threads>>>(m_data->d_particles, m_particleCount, stepDt);
            cudaDeviceSynchronize();

            ComputeKeysKernel<<<blocks, threads>>>(
                m_data->d_particles, m_data->d_spatialKeys, m_data->d_sortBuffer, m_data->d_sortedIndices, m_particleCount);
            cudaDeviceSynchronize();

            thrust::device_ptr<uint32_t> sort_ptr(m_data->d_sortBuffer);
            thrust::device_ptr<uint32_t> idx_ptr(m_data->d_sortedIndices);
            thrust::sort_by_key(sort_ptr, sort_ptr + m_particleCount, idx_ptr);

            cudaMemset(m_data->d_spatialOffsets, 0xFF, m_particleCount * sizeof(uint32_t));
            MarkOffsetsKernel<<<blocks, threads>>>(m_data->d_sortBuffer, m_data->d_spatialOffsets, m_particleCount);
            cudaDeviceSynchronize();

            CalculateDensitiesKernel<<<blocks, threads>>>(
                m_data->d_particles, m_data->d_spatialKeys, m_data->d_spatialOffsets, m_data->d_sortedIndices, m_particleCount);
            cudaDeviceSynchronize();

            CalculatePressureKernel<<<blocks, threads>>>(
                m_data->d_particles, m_data->d_spatialKeys, m_data->d_spatialOffsets, m_data->d_sortedIndices, m_particleCount, stepDt);
            cudaDeviceSynchronize();

            CalculateViscosityKernel<<<blocks, threads>>>(
                m_data->d_particles, m_data->d_spatialKeys, m_data->d_spatialOffsets, m_data->d_sortedIndices, m_particleCount, stepDt);
            cudaDeviceSynchronize();

            UpdatePositionsKernel<<<blocks, threads>>>(m_data->d_particles, m_particleCount, stepDt);
            cudaDeviceSynchronize();
        }

        // copy to VBO
        VBOParticle* d_vbo = nullptr;
        size_t size = 0;
        cudaGraphicsMapResources(1, &m_data->d_vboResource);
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &size, m_data->d_vboResource);
        CopyToVBOKernel<<<blocks, threads>>>(m_data->d_particles, d_vbo, m_particleCount);
        cudaGraphicsUnmapResources(1, &m_data->d_vboResource);
    }

    void Simulation::RegisterGLBuffer(GLuint buffer)
    {
        cudaGraphicsGLRegisterBuffer(&m_data->d_vboResource, buffer, cudaGraphicsMapFlagsWriteDiscard);
    }

    void Simulation::InitSpawnParticles()
    {
        std::vector<dim3::Particle> h(m_maxParticles);

        int cols  = (int)cbrtf((float)m_particleCount);
        int rows  = cols;
        int depth = (m_particleCount + cols * rows - 1) / (cols * rows);

        float startX = -(cols  * m_config.spawnSpacing) * 0.5f;
        float startY =  (rows  * m_config.spawnSpacing) * 0.5f;
        float startZ = -(depth * m_config.spawnSpacing) * 0.5f;

        for (int idx = 0; idx < m_particleCount; idx++)
        {
            int x = idx % cols;
            int y = (idx / cols) % rows;
            int z = idx / (cols * rows);

            h[idx].position = make_float3(
                startX + x * m_config.spawnSpacing,
                startY - y * m_config.spawnSpacing,
                startZ + z * m_config.spawnSpacing
            );
            h[idx].predictedPosition = h[idx].position;
            h[idx].velocity     = make_float3(0.0f, 0.0f, 0.0f);
            h[idx].acceleration = make_float3(0.0f, 0.0f, 0.0f);
            h[idx].density      = m_config.targetDensity;
            h[idx].nearDensity  = 0.0f;
            h[idx].pressure     = 0.0f;
            h[idx].nearPressure = 0.0f;
            h[idx].mass         = 1.0f;
        }

        cudaMemcpy(m_data->d_particles, h.data(), m_maxParticles * sizeof(Particle), cudaMemcpyHostToDevice);
    }
}

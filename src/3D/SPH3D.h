#pragma once
#include "../SPH.h"
#include "Spawner3D.h"

namespace sph::dim3
{
    class Simulation : public ISimulation
    {
    public:
        Simulation(int initialParticles, int maxParticles = 10000);
        Simulation(const SpawnData& spawnData, int maxParticles = 10000);
        ~Simulation() override;

        void Update(float dt) override;

        void UpdateConfig(const Config& config) override;

        int ParticleCount() const override { return m_particleCount; }
        const Config& GetConfig() const override { return m_config; }

        void RegisterGLBuffer(GLuint buffer) override;

    private:
        void InitSpawnParticles();

    private:
        Config m_config;

        const int m_maxParticles;
        int m_particleCount;

        struct DeviceData;
        DeviceData* m_data;
    };
}
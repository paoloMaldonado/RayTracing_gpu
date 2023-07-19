#if !defined(__PLASTIC_CUH__)
#define __PLASTIC_CUH__

#include "core/material.cuh"
#include "core/surfaceInteraction.cuh"
#include "core/dependencies.cuh"

class PlasticMaterial : public Material
{
public:
	vec3 Kd;
	vec3 Ks;
	float shininess;
	bool blinn;

	PlasticMaterial() = default;
	__device__
	PlasticMaterial(const vec3& Kd, const vec3& Ks, const float& shininess, const bool blinn = 0);
	__device__
	virtual ~PlasticMaterial() {}
	__device__
	virtual void compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer) override;
};

#endif
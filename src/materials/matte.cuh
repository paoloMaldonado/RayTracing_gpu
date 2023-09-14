#if !defined(__MATTE_CUH__)
#define __MATTE_CUH__

#include "core/material.cuh"
#include "core/surfaceInteraction.cuh"
#include "core/dependencies.cuh"

class MatteMaterial : public Material
{
public:
	Spectrum Kd;

	MatteMaterial() = default;
	__device__
	MatteMaterial(const Spectrum& Kd);
	__device__
	virtual ~MatteMaterial() {}
	__device__
	virtual void compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer) override;
};

#endif
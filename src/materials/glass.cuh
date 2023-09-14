#if !defined(__GLASS_CUH__)
#define __GLASS_CUH__

#include "core/material.cuh"
#include "core/surfaceInteraction.cuh"
#include "core/dependencies.cuh"

class GlassMaterial : public Material
{
public:
	Spectrum Ks;
	Spectrum Kt;
	float eta;

	GlassMaterial() = default;
	__device__
	GlassMaterial(const Spectrum& Ks, const Spectrum& Kt, const float& eta);
	__device__
	virtual ~GlassMaterial() {}
	__device__
	virtual void compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer) override;
};

#endif



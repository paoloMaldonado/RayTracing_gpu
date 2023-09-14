#if !defined(__MIRROR_CUH__)
#define __MIRROR_CUH__

#include "core/material.cuh"
#include "core/surfaceInteraction.cuh"
#include "core/dependencies.cuh"

class MirrorMaterial : public Material
{
public:
	Spectrum Ks;

	MirrorMaterial() = default;
	__device__
	MirrorMaterial(const Spectrum & Ks);
	__device__
	virtual ~MirrorMaterial() {}
	__device__
	virtual void compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer) override;
};


#endif
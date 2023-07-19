#if !defined(__MATERIAL_CUH__)
#define __MATERIAL_CUH__

#include "dependencies.cuh"

class Material
{
public:
	Material() = default;
	__device__
	virtual ~Material() {}
	__device__
    virtual void compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer) = 0;
};

#endif
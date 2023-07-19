#if !defined(__SURFACEINTERACTION_CUH__)
#define __SURFACEINTERACTION_CUH__

#include "memory/memory.cuh"
#include "reflection.cuh"
#include "shape.cuh"

class SurfaceInteraction
{
public:
	float t;
	Shape* hitobject;
	BSDF bsdf;
	vec3 wo, n;
	vec3 p;

	__device__
	SurfaceInteraction();
	__device__
	SurfaceInteraction(const vec3& p, const vec3& wo, const vec3& n);
	__device__
	void compute_scattering_functions(MemoryManager& mem_buffer);

};

#endif
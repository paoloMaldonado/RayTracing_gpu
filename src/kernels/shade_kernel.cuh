#if !defined(__SHADE_KERNEL_CUH__)
#define __SHADE_KERNEL_CUH__

#include "core/surfaceInteraction.cuh"

__device__
inline vec3 shade(SurfaceInteraction& si, const vec3& wi, MemoryManager& mem_buffer, const bool& visibility)
{
	// if the point is in shadow, returns black
	if (visibility)
		return vec3(0.0f);

	si.compute_scattering_functions(mem_buffer);  // surfaceInteraction instance will DO change

	//vec3 wi = wi;
	vec3 wo = si.wo;
	vec3 n = si.n;

	if (dot(n, wo) < 0.0f) n = -n;
	vec3 I = vec3(1.0f, 1.0f, 1.0f);

	vec3 L = si.bsdf.f(wi, wo) * I * fmaxf(0.0f, dot(n, wi));
	return L;
}

#endif
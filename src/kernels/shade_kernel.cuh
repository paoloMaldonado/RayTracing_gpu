#if !defined(__SHADE_KERNEL_CUH__)
#define __SHADE_KERNEL_CUH__

#include "core/surfaceInteraction.cuh"

__device__
inline vec3 shade(const SurfaceInteraction& si, const vec3& wi, const bool& visibility)
{
	// if the point is in shadow, returns black
	if (visibility)
		return vec3(0.0f);

	vec3 wo = si.wo;
	vec3 n = si.n;

	n = dot(n, wo) < 0.0f ? -n : n;
	vec3 I = vec3(1.0f, 1.0f, 1.0f);

	vec3 L = si.bsdf.f(wi, wo) * I * fmaxf(0.0f, dot(n, wi));
	return L;
}

#endif
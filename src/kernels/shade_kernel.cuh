#if !defined(__SHADE_KERNEL_CUH__)
#define __SHADE_KERNEL_CUH__

#include "core/surfaceInteraction.cuh"

__device__
inline Spectrum shade(const SurfaceInteraction& si, const vec3& wi, const bool& visibility)
{
	// if the point is in shadow, returns black
	if (visibility)
		return Spectrum(0.0f);

	vec3 wo = si.wo;
	normal3 n = si.n;

	n = dot(n, wo) < 0.0f ? -n : n;
	Spectrum I = Spectrum(1.0f, 1.0f, 1.0f);

	Spectrum L = si.bsdf.f(wi, wo) * I * fmaxf(0.0f, dot(n, wi));
	return L;
}

#endif
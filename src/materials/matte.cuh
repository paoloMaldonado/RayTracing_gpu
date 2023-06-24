#if !defined(__MATTE_CUH__)
#define __MATTE_CUH__

#include "misc/vec3.cuh"
#include "core/surfaceInteraction.cuh"

class MatteMaterial
{
public:
	vec3 Kd;

	MatteMaterial() = default;
	__device__
    MatteMaterial(const vec3& Kd) : Kd(Kd)
	{}
	__device__
	__inline__ void computeScatteringFunctions(SurfaceInteraction& si)
	{
		return;
	}
};

#endif
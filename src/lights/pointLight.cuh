#if !defined(__POINT_LIGHT_CUH__)
#define __POINT_LIGHT_CUH__

#include "core/surfaceInteraction.cuh"
#include "core/ray.cuh"

class PointLight
{
public:
	point3 pos;
	
	PointLight() = default;
	__device__
	PointLight(const point3& position) : pos(position)
	{}
	__device__
	Spectrum sample_li(const SurfaceInteraction& rec, const VisibilityTester& visibility, vec3& wi, bool& in_shadow) const
	{
		point3 p = rec.p;
		wi = normalize(pos - p);
		Ray shadow_ray = Ray(p + wi*0.0001f, wi);
		in_shadow = visibility.test_shadow(shadow_ray);

		return Spectrum(1.0f);  // white light
	}
};

#endif
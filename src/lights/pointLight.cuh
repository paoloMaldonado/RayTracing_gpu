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

		normal3 n = rec.n; 
		n = dot(n, rec.wo) < 0.0f ? -n : n;
		n = n * 0.01f;

		Ray shadow_ray = Ray(p + n.to_vec3(), wi);
		in_shadow = visibility.test_shadow(shadow_ray);

		return Spectrum(1.0f);  // white light
	}
};

#endif
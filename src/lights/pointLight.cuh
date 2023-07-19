#if !defined(__POINT_LIGHT_CUH__)
#define __POINT_LIGHT_CUH__

#include "core/surfaceInteraction.cuh"
#include "core/ray.cuh"

class PointLight
{
public:
	vec3 pos;
	
	PointLight() = default;
	__device__
	PointLight(const vec3& position) : pos(position)
	{}
	__device__
	vec3 sample_li(const SurfaceInteraction& rec, const Ray& ray, const VisibilityTester& visibility, vec3& wi, bool& in_shadow) const
	{
		vec3 p = ray.point_at_parameter(rec.t);
		wi = normalize(pos - p);
		Ray shadow_ray = Ray(p + wi*0.0001f, wi);
		in_shadow = visibility.test_shadow(shadow_ray);

		return vec3(1.0f);  // white light
	}
};

#endif
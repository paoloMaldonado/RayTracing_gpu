#if !defined(__RAY_CUH__)
#define __RAY_CUH__

#include "geometry.cuh"

class Ray
{
public:
	point3 origin;
	vec3 direction;

	Ray() = default;
	__device__ 
	Ray(const point3& origin, const vec3& direction) : origin(origin), direction(direction)
	{}
	__device__
	inline point3 point_at_parameter(const float& t) const { return origin + t * direction; }
};

#endif
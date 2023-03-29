#if !defined(__RAY_CUH__)
#define __RAY_CUH__

#include "vec3.cuh"

class Ray
{
public:
	vec3 origin;
	vec3 direction;

	Ray() = default;
	__device__ 
	Ray(const vec3& origin, const vec3& direction) : origin(origin), direction(direction)
	{}
	__device__
	inline vec3 point_at_parameter(const float& t) { return origin + t*direction; }
};

#endif
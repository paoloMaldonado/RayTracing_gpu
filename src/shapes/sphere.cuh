#if !defined(__SPHERE_CUH__)
#define __SPHERE_CUH__

#include "core/ray.cuh"
#include "core/shape.cuh"
#include <math.h>

class Sphere : public Shape
{
public:
	vec3 center;
	float radius;

	Sphere() = default;
	__device__
	Sphere(const vec3& c, const float& r, Material* mat) : center(c), radius(r), Shape(mat)
	{}
	__device__ 
	virtual bool hitted_by(const Ray& ray, float& t) const override;
	__device__ 
	virtual vec3 compute_normal_at(const vec3& p) const override { return (p - center)/radius; }
};

#endif

#if !defined(__SPHERE_CUH__)
#define __SPHERE_CUH__

#include "core/ray.cuh"
#include "core/shape.cuh"
#include <math.h>

class Sphere : public Shape
{
public:
	point3 center;
	float radius;

	Sphere() = default;
	__device__
	Sphere(const point3& c, const float& r, Material* mat) : center(c), radius(r), Shape(mat)
	{}
	__device__ 
	virtual bool hitted_by(const Ray& ray, float& t) const override;
	__device__ 
	virtual normal3 compute_normal_at(const point3& p) const override 
	{ 
		vec3 n = (p - center)/radius;
		return normal3(n.x, n.y, n.z);
	}
};

#endif

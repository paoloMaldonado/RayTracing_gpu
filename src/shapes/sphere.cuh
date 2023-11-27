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
	Sphere(const point3& c, const float& r) : center(c), radius(r)
	{}
	__device__ 
	virtual bool hitted_by(const Ray& ray, float& t, float& u, float& v) const override;
	__device__
	virtual normal3 compute_normal_at(const point3& p, const float& u = 0.0f, const float& v = 0.0f) const override
	{ 
		vec3 n = (p - center)/radius;
		return normal3(n.x, n.y, n.z);
	}
	__device__
	virtual Shape* get_shape() override { return this; }
};

__device__
Shape* createSphereShape(const point3& center, const float& radius);

#endif

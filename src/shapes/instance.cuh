#if !defined(__INSTANCE_CUH__)
#define __INSTANCE_CUH__

#include "core/ray.cuh"
#include "core/shape.cuh"
#include "core/transform.cuh"
#include <math.h>

class Instance
{
public:
	Shape* object_ptr;
	Transform inv_matrix;

	Instance() = default;
	__device__
	Instance(Shape* object_ptr);

	__device__
	Instance(Shape* object_ptr, const Transform& transform);

	__device__
	bool hitted_by(const Ray& ray, float& t, Ray& inv_ray) const;
	__device__
	normal3 compute_normal(const point3& p) const;
};


#endif
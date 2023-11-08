#if !defined(__COMPOUND_CUH__)
#define __COMPOUND_CUH__

#include "core/ray.cuh"
#include "core/shape.cuh"
#include <math.h>


class Compound : Shape
{
public:
	Shape** objects;
	Shape* isect_object;  // intersected object (if any)
	unsigned int nObjects;

	Compound() = default;
	__device__
	Compound(Shape** objects, const unsigned int& nObjects) : 
		objects(objects), nObjects(nObjects), isect_object(nullptr)
	{}
	__device__
	virtual bool hitted_by(const Ray & ray, float& t) const override;
	__device__
	virtual normal3 compute_normal_at(const point3 & p) const override;
	__device__
	virtual Shape* get_shape() override { return isect_object; }
};


#endif
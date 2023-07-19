#if !defined(__SHAPE_CUH__)
#define __SHAPE_CUH__

#include "dependencies.cuh"

class Shape
{
public:
	Material* material;

	Shape() = default;
	__device__
	Shape(Material* mat);
	__device__ 
	virtual bool hitted_by(const Ray& ray, float& t) const = 0;
	__device__
	virtual vec3 compute_normal_at(const vec3& p) const = 0;
	__device__
	void compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer);
};

#endif

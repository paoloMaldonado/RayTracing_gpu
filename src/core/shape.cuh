#if !defined(__SHAPE_CUH__)
#define __SHAPE_CUH__

#include "dependencies.cuh"

class Shape
{
public:
	Material* material;

	__device__
	Shape() : material(nullptr)
	{}
	__device__
	Shape(Material* mat);
	__device__ 
	virtual bool hitted_by(const Ray& ray, float& t, float& u, float& v) const = 0;
	__device__
	virtual normal3 compute_normal_at(const point3& p, const float& u = 0.0f, const float& v = 0.0f) const = 0;
	__device__
	void compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer);
	__device__
	virtual Shape* get_shape() = 0;
	__device__
	void add_material(Material* material) { this->material = material; };
};

#endif

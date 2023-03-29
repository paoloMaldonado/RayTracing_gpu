#if !defined(__SPHERE_CUH__)
#define __SPHERE_CUH__

#include "misc/ray.cuh"
#include "materials/material.cuh"
#include <math.h>

class Sphere
{
public:
	vec3 center;
	float radius;
	Material material;

	Sphere() = default;
	__host__ __device__
	Sphere(const vec3& c, const float& r, Material mat) : center(c), radius(r), material(mat)
	{}
	__device__ bool hitted_by(const Ray& ray, float& t) const;
	__device__ 
	inline vec3 compute_normal_at(const vec3& p) const { return (p - center)/radius; }
	
	__host__ __device__
	inline vec3 color() const { return material.color; }
	__host__ __device__
	inline vec3 specular_color() const { return material.specular_color; }
	__host__ __device__
	inline float shininess() const { return material.shininess; }
};

#endif

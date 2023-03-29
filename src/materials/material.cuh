#if !defined(__MATERIAL_CUH__)
#define __MATERIAL_CUH__

#include "misc/vec3.cuh"

class Material
{
public:
	vec3 color;
	vec3 specular_color;
	float shininess;

	Material() = default;
	__host__ __device__ 
	Material(vec3 color, vec3 specular_color, float shininess) : 
		color(color), specular_color(specular_color), shininess(shininess)
	{}
};

#endif
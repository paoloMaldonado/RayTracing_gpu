#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "dependencies.cuh"

class Scene
{
public:
	Shape** d_objects;				   // GPU (device pointer)
	Material** d_materials;
	unsigned int primitive_count;
	unsigned int material_count;

	Scene() = default;
	Scene(Shape** list_objects, const unsigned int& N, Material** list_materials, const unsigned int& nMat);
	void build();
	void destroy();
};




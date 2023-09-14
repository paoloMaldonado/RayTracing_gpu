#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "dependencies.cuh"

class Scene
{
public:
	Shape** d_shapes;			       // explicit shapes (device pointer)
	Instance** d_instances;			   // GPU (device pointer) - instancing
	Material** d_materials;
	unsigned int instance_count;
	unsigned int primitive_count;
	unsigned int material_count;

	Scene() = default;
	Scene(Shape** list_shapes, Instance** list_instances, const unsigned int& N_obj, const unsigned int& N, Material** list_materials, const unsigned int& nMat);
	void build();
	void destroy();
};




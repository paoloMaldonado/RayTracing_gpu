#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "dependencies.cuh"

class Scene
{
public:
	Shape** d_shapes;				   // explicit shapes (device pointer)
	Instance** d_instances;		       // GPU (device pointer) - instancing
	Material** d_materials;
	unsigned int nPrimitives;
	unsigned int nMaterials;

	Scene() = default;
	Scene(const unsigned int& nPrimitives, const unsigned int& nMaterials);
	//void makeShapes(Shape** shapes, const unsigned int& nPrimitives);
	void build();
	void destroy();
};




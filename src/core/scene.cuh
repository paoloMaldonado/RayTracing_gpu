#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "render/Loader.h"

#include "dependencies.cuh"

class Scene
{
public:
	Scene() = default;
	Scene(const unsigned int& nPrimitives, const unsigned int& nMaterials);
	Scene(const unsigned int& nMaterials);
	void load_obj_to_gpu(std::string inputfile, std::string materialpath);
	void build();
	void destroy();

	inline int number_of_primitives() { return nPrimitives; }
	inline Instance** get_pointer_to_instances() { return d_instances; }

private:
	Shape** d_shapes;				   // explicit shapes (device pointer)
	Instance** d_instances;		       // GPU (device pointer) - instancing
	Material** d_materials;
	Transform** d_transforms;
	unsigned int nPrimitives;
	unsigned int nMaterials;

	// OBJ to GPU
	Loader obj_loader;
	int* d_indices;
	point3* d_vertices;
	MatParam* d_material_params;
	int* d_triangles_per_shape;
	int nObjectsInMesh;

};

__device__
Shape** makeShapes(const char* name, const unsigned int nPrimitives, int* indices = nullptr, point3* vertices = nullptr);
__device__
Material* makeMaterial(const MaterialType& material);




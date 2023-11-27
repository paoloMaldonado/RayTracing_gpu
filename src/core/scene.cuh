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
	void build(bool demo = false);
	void destroy();

	inline int number_of_primitives() { return nPrimitives; }
	inline Instance** get_pointer_to_instances() { return d_instances; }

private:
	Instance** d_instances;		       // GPU (device pointer) - instancing
	Material** d_materials;
	Transform** d_transforms;
	unsigned int nPrimitives;
	unsigned int nMaterials;

	// OBJ to GPU
	Loader obj_loader;
	int* d_indices;
	int* d_indices_normal;
	point3* d_vertices;
	normal3* d_normals;
	MatParam* d_material_params;
	int* d_triangles_per_shape;
	int nObjectsInMesh;

};

__device__
Shape** makeShapes(const char* name, const unsigned int nPrimitives, int* indices = nullptr, 
	               int* indices_normal = nullptr, point3* vertices = nullptr, normal3* normals = nullptr);
__device__
Material* makeMaterial(const MatParam& material_parameters);




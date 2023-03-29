#include "Scene.h"

Scene::Scene(const std::vector<Sphere>& list_objects) : list_objects{list_objects}, d_objects(nullptr)
{
	primitive_count = list_objects.size();
}

void Scene::build()
{
	unsigned int vector_size = list_objects.size() * sizeof(Sphere);
	cudaMalloc(&d_objects, vector_size);
	cudaMemcpy(d_objects, list_objects.data(), vector_size, cudaMemcpyHostToDevice);
}

void Scene::destroy()
{
	cudaFree(d_objects);
}


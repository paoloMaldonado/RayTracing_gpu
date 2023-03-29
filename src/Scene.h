#pragma once

#include <vector>
#include "shapes/sphere.cuh"

class Scene
{
public:
	std::vector<Sphere> list_objects;  // CPU
	Sphere* d_objects;				   // GPU (device pointer)
	unsigned int primitive_count;

	Scene() = default;
	Scene(const std::vector<Sphere>& list_objects);
	void build();
	void destroy();
};


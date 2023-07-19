#include "shape.cuh"
#include "material.cuh"
#include <device_launch_parameters.h>

__device__
Shape::Shape(Material* mat) : material(mat)
{}

__device__
void Shape::compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer)
{
	material->compute_scattering_functions(si, mem_buffer);
}


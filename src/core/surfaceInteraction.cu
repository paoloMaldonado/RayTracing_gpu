#include "surfaceInteraction.cuh"

__device__
SurfaceInteraction::SurfaceInteraction() : t(INFINITY), hitobject(nullptr)
{}

__device__
SurfaceInteraction::SurfaceInteraction(const vec3& p, const vec3& wo, const vec3& n) :
	t(INFINITY), hitobject(nullptr), p(p), wo(wo), n(n)
{}

__device__
void SurfaceInteraction::compute_scattering_functions(MemoryManager& mem_buffer)
{
	hitobject->compute_scattering_functions(this, mem_buffer);
}
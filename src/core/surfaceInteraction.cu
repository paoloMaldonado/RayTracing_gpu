#include "surfaceInteraction.cuh"

__device__
SurfaceInteraction::SurfaceInteraction() : t(INFINITY), hitobject(nullptr), p(0.0f), wo(0.0f), n(0.0f)  // initialize all in zero
{}

__device__
SurfaceInteraction::SurfaceInteraction(const point3& p, const vec3& wo, const normal3& n) :
	t(INFINITY), hitobject(nullptr), p(p), wo(wo), n(n)
{}

__device__
void SurfaceInteraction::compute_scattering_functions(MemoryManager& mem_buffer)
{
	hitobject->compute_scattering_functions(this, mem_buffer);
}
#include "matte.cuh"
#include "memory/memory.cuh"

__device__
MatteMaterial::MatteMaterial(const vec3& Kd) : Kd(Kd)
{}

__device__
void MatteMaterial::compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer)
{	
	si->bsdf.add(ALLOC(mem_buffer, LambertianReflection)(Kd));
}
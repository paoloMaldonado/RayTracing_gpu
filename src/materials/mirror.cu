#include "mirror.cuh"
#include "memory/memory.cuh"

__device__
MirrorMaterial::MirrorMaterial(const vec3& Ks) : Ks(Ks)
{}

__device__
void MirrorMaterial::compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer)
{
	vec3 normal = si->n;

	Fresnel* fresnel = ALLOC(mem_buffer, FresnelNoOp)();
	si->bsdf.add(ALLOC(mem_buffer, SpecularReflection)(Ks, normal, fresnel));
}
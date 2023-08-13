#include "glass.cuh"
#include "memory/memory.cuh"

__device__
GlassMaterial::GlassMaterial(const vec3& Ks, const vec3& Kt, const float& eta) : Ks(Ks), Kt(Kt), eta(eta)
{}

__device__
void GlassMaterial::compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer)
{
	vec3 normal = si->n;

	if (Ks.isBlack() && Kt.isBlack()) return;

	Fresnel* fresnel = ALLOC(mem_buffer, FresnelDielectric)(1.0f, eta);
	if (!Ks.isBlack())
	{
		si->bsdf.add(ALLOC(mem_buffer, SpecularReflection)(Ks, normal, fresnel));
	}
		
	if (!Kt.isBlack())
	{
		si->bsdf.add(ALLOC(mem_buffer, SpecularRefraction)(Kt, normal, 1.0f, eta));
	}
		
}
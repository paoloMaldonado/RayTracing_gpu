#include "plastic.cuh"

__device__
PlasticMaterial::PlasticMaterial(const vec3& Kd, const vec3& Ks, const float& shininess, const bool blinn) :
	Kd(Kd), Ks(Ks), shininess(shininess), blinn(blinn)
{}

__device__
void PlasticMaterial::compute_scattering_functions(SurfaceInteraction* si, MemoryManager& mem_buffer)
{
	vec3 normal = si->n;
	si->bsdf.add(ALLOC(mem_buffer, LambertianReflection)(Kd));
	if(blinn)
		si->bsdf.add(ALLOC(mem_buffer, BlinnPhongReflection)(Ks, shininess, normal));
	else
		si->bsdf.add(ALLOC(mem_buffer, PhongReflection)(Ks, shininess, normal));
}

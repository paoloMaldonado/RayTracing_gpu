#include "reflection.cuh"

__device__
BSDF::~BSDF()
{
	for (int i = 0; i < num_bxdfs; ++i)
	{
		if (bxdfs[i])
		{
			bxdfs[i]->~BxDF();
		}
			
	}
}

__device__
vec3 BSDF::f(const vec3& wi, const vec3& wo) const
{
	vec3 f(0.0f);
	for (int i = 0; i < num_bxdfs; ++i)
	{
		f += bxdfs[i]->f(wi, wo);
	}
	return f;
}
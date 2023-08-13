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
	num_bxdfs = 0;
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

__device__ 
vec3 BSDF::sample_f(const vec3& wo, vec3& wi, const BxDFType& type) const
{
	vec3 f(0.0f);
	BxDF* bxdf = nullptr;
	for (int i = 0; i < num_bxdfs; ++i)
	{
		if (bxdfs[i]->type == type)
		{
			bxdf = bxdfs[i];
			break;
		}	
	}
	if (bxdf != nullptr) { f = bxdf->sample_f(wo, wi); }
	return f;
}

__device__ 
vec3 refract(const vec3& n, const vec3& wi, const float& etaRatio)
{
	float cosThetaI = dot(wi, n);
	// compute cosThetaT using Snell's Law
	float sin2ThetaI = fmax(0.0f, 1.0f - cosThetaI * cosThetaI);
	float sin2ThetaT = etaRatio * etaRatio * sin2ThetaI;
	// check for total internal reflection (TIR)
	if (sin2ThetaT >= 1.0f)
		return vec3(0.0f); // no refraction
	
	float cosThetaT = sqrtf(1.0f - sin2ThetaT);

	return etaRatio * -wi + (etaRatio * cosThetaI - cosThetaT) * n;
}

__device__
float reflectanceFresnel(float cosThetaI, float etaI, float etaT)
{
	// determine which index is incident and which is refracted
	bool entering = cosThetaI > 0.0f;
	if (!entering)
	{
		swap(etaI, etaT);
		cosThetaI = fabs(cosThetaI);
	}
	// compute cosThetaT using Snell's Law
	float sinThetaI = sqrtf(fmax(0.0f, 1.0f - (cosThetaI * cosThetaI)));
	float sinThetaT = (etaI / etaT) * sinThetaI;
	// check for total internal reflection (TIR)
	if (sinThetaT >= 1.0f)
		return 1.0f; // no refraction

	float costThetaT = sqrtf(fmax(0.0f, 1 - (sinThetaT * sinThetaT)));

	float Rperp = (etaI * cosThetaI - etaT * costThetaT) /
				  (etaI * cosThetaI + etaT * costThetaT);
	float Rparl = (etaT * cosThetaI - etaI * costThetaT) /
				  (etaT * cosThetaI + etaI * costThetaT);

	return (Rperp * Rperp + Rparl * Rparl) / 2.0f;
}

__device__ float FresnelDielectric::evaluate(float cosThetaI) const
{
	return reflectanceFresnel(cosThetaI, etaI, etaT);
}


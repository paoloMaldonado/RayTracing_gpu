#if !defined(__REFLECTION_CUH__)
#define __REFLECTION_CUH__

#include "geometry.cuh"
#include "utility/utils.cuh"
#include <math.h>

__device__
inline vec3 reflect(const normal3& n, const vec3& wi) { return normalize((2.0f * n * (dot(n, wi))) - wi); }
__device__
bool refract(const normal3& n, const vec3& wi, const float& etaRatio, vec3& wo);
__device__
float reflectanceFresnel(float cosThetaI, float etaI, float etaT);

enum class BxDFType
{
	SPECULAR_REFLECTION = 0,
	SPECULAR_REFRACTION = 1,
	REFLECTION          = 2,
	DIFFUSE             = 3
};

class BxDF
{
public:
	BxDFType type;
	BxDF() = default;
	__device__
	BxDF(const BxDFType& type): type(type)
	{}
	__device__
	virtual ~BxDF() {}
	__device__
	virtual Spectrum f(const vec3& wi, const vec3& wo) const = 0;
	__device__
	virtual Spectrum sample_f(const vec3& wo, vec3& wi) const = 0;
};

class Fresnel
{
public:
	__device__
	virtual ~Fresnel() {}
	__device__
	virtual	float evaluate(float cosThetaI) const = 0;
};

class FresnelDielectric : public Fresnel
{
public:
	float etaI;
	float etaT;

	__device__
	FresnelDielectric(float etaI, float etaT) : etaI(etaI), etaT(etaT)
	{}
	__device__
	virtual	float evaluate(float cosThetaI) const;

};

class FresnelNoOp : public Fresnel
{
public:
	__device__
	virtual	float evaluate(float cosThetaI) const { return 1.0f; }
};

class LambertianReflection : public BxDF
{
public:
	Spectrum Kd;

	LambertianReflection() = default;
	__device__
	LambertianReflection(const Spectrum& Kd) : Kd(Kd), BxDF(BxDFType::DIFFUSE)
	{}
	__device__
	virtual ~LambertianReflection() {}
	__device__
	virtual Spectrum f(const vec3& wi, const vec3& wo) const override
	{
		return (Kd * invPi());
	};
	__device__
	virtual Spectrum sample_f(const vec3& wo, vec3& wi) const override { return Spectrum(0.0f); };
};

class PhongReflection : public BxDF
{
public:
	Spectrum Ks;
	float pn;  // specular phong exponent
	normal3 n;    // normal vector

	PhongReflection() = default;
	__device__
	PhongReflection(const Spectrum& Ks, 
					const float& phong_exponent, const normal3& n) : 
		Ks(Ks), pn(phong_exponent), n(n), BxDF(BxDFType::REFLECTION)
	{}
	__device__
	virtual ~PhongReflection() {}
	__device__
	virtual Spectrum f(const vec3& wi, const vec3& wo) const override
	{
		float cos_alpha = fmaxf(0.0f, dot(reflect(n, wi), wo));  // =cos(alpha) where alpha is the angle between perfect reflection and wo                         
		float norm_factor = (pn + 2.0f) / twopi();
		
		return Ks * norm_factor * powf(cos_alpha, pn);
	}
	__device__
	virtual Spectrum sample_f(const vec3& wo, vec3& wi) const override { return Spectrum(0.0f); };
};

class BlinnPhongReflection : public BxDF
{
public:
	Spectrum Ks;
	float pn;  // specular phong exponent
	normal3 n;    // normal vector

	BlinnPhongReflection() = default;
	__device__
	BlinnPhongReflection(const Spectrum& Ks,
						 const float& phong_exponent, const normal3& n) :
		Ks(Ks), pn(phong_exponent), n(n), BxDF(BxDFType::REFLECTION)
	{}
	__device__
	virtual ~BlinnPhongReflection() {}
	__device__
	virtual Spectrum f(const vec3& wi, const vec3& wo) const override
	{
		vec3 h = normalize(wi + wo);
		float cos_alpha = fmaxf(0.0f, dot(n, h));  // =cos(alpha) where alpha is the angle between half vector and normal                         
		float norm_factor = ((pn + 2.0f) * (pn + 4.0f)) / 
							((8.0f * pi()) * (powf(2, -pn / 2.0f) + pn));

		return Ks * norm_factor * powf(cos_alpha, pn);
	}
	__device__
	virtual Spectrum sample_f(const vec3& wo, vec3& wi) const override { return Spectrum(0.0f); };
};

class SpecularReflection : public BxDF
{
public:
	Spectrum R;		    // scale factor
	normal3 n;			// normal vector
	Fresnel* fresnel;   // fresnel reflectance factor

	SpecularReflection() = default;
	__device__
	SpecularReflection(const Spectrum& R, const normal3& n, Fresnel* fresnel) : 
		R(R), n(n), fresnel(fresnel), BxDF(BxDFType::SPECULAR_REFLECTION)
	{}
	__device__
	virtual ~SpecularReflection() {}
	__device__
	virtual Spectrum f(const vec3& wi, const vec3& wo) const override
	{
		return Spectrum(0.0f);
	}
	__device__
	virtual Spectrum sample_f(const vec3& wo, vec3& wi) const override
	{
		// Generate incident ray with perfect specular direction
		normal3 custom_n = dot(n, wo) < 0.0f ? -n : n;
		wi = reflect(custom_n, wo); 
		// Compute the angle between the reflected vector and the normal
		float cosThetaI = dot(wi, n);

		return fresnel->evaluate(cosThetaI) * R / fabs(cosThetaI);
	}
};

class SpecularRefraction : public BxDF
{
public:
	Spectrum T;					  // scale factor
	normal3 n;						  // normal vector
	float etaA;					 	
	float etaB;
	FresnelDielectric fresnel;    // fresnel reflectance factor

	SpecularRefraction() = default;
	__device__
	SpecularRefraction(const Spectrum& T, const normal3& n, float etaA, float etaB) :
		T(T), n(n), etaA(etaA), etaB(etaB), fresnel(etaA, etaB), BxDF(BxDFType::SPECULAR_REFRACTION)
	{}
	__device__
	virtual ~SpecularRefraction() {}
	__device__
	virtual Spectrum f(const vec3& wi, const vec3& wo) const override
	{
		return Spectrum(0.0f);
	}
	__device__
	virtual Spectrum sample_f(const vec3& wo, vec3& wi) const override
	{
		// determine which index is incident and which is refracted
		float cosThetaO = dot(n, wo);
		bool entering = cosThetaO > 0.0f;
		float etaI = entering ? etaA : etaB;
		float etaT = entering ? etaB : etaA;
		
		// Generate refracted ray with specular direction
		normal3 custom_n = dot(n, wo) < 0.0f ? -n : n;
		if (!refract(custom_n, wo, etaI / etaT, wi)) return Spectrum(0.0f);

		// Compute the angle between refracted direction and the normal
		float cosThetaI = dot(wi, n);

		Spectrum Kt = (1.0f - fresnel.evaluate(cosThetaI)) * T;
		return Kt / fabs(cosThetaI);
	}
};

class BSDF
{
public:
	int num_bxdfs = 0;
	BxDF* bxdfs[4];
	
	BSDF() = default;
	__device__
	~BSDF();
	__device__
	void add(BxDF* bxdf) 
	{ 
		bxdfs[num_bxdfs++] = bxdf; 
	};
	__device__
	Spectrum f(const vec3& wi, const vec3& wo) const;
	__device__
	Spectrum sample_f(const vec3& wo, vec3& wi, const BxDFType& type) const;
	__device__
	void clear() { this->~BSDF(); }
};

#endif

#if !defined(__REFLECTION_CUH__)
#define __REFLECTION_CUH__

#include "vec3.cuh"
#include "utility/utils.cuh"
#include <math.h>

__device__
inline vec3 reflect(const vec3& n, const vec3& wi) { return normalize((2.0f * n * (dot(n, wi))) - wi); }

class BxDF
{
public:
	vec3 n;
	__device__
	BxDF() { n.x = n.y = n.z = 0.0f; };
	__device__
	BxDF(const vec3& n): n(n)
	{}
	__device__
	virtual ~BxDF() {}
	__device__
	virtual vec3 f(const vec3& wi, const vec3& wo) const = 0;
};

class LambertianReflection : public BxDF
{
public:
	vec3 Kd;

	LambertianReflection() = default;
	__device__
	LambertianReflection(const vec3& Kd) : Kd(Kd)
	{}
	__device__
	virtual ~LambertianReflection() {}
	__device__
	virtual vec3 f(const vec3& wi, const vec3& wo) const override
	{
		return (Kd * invPi());
	};
};

class PhongReflection : public BxDF
{
public:
	vec3 Ks;
	float pn;  // specular phong exponent
	vec3 n;    // normal vector

	PhongReflection() = default;
	__device__
	PhongReflection(const vec3& Ks, 
					const float& phong_exponent, const vec3& n) : 
		Ks(Ks), pn(phong_exponent), n(n)
	{}
	__device__
	virtual ~PhongReflection() {}
	__device__
	virtual vec3 f(const vec3& wi, const vec3& wo) const override
	{
		float cos_alpha = fmaxf(0.0f, dot(reflect(n, wi), wo));  // =cos(alpha) where alpha is the angle between perfect reflection and wo                         
		float norm_factor = (pn + 2.0f) / twopi();
		
		return Ks * norm_factor * powf(cos_alpha, pn);
	}
};

class BlinnPhongReflection : public BxDF
{
public:
	vec3 Ks;
	float pn;  // specular phong exponent
	vec3 n;    // normal vector

	BlinnPhongReflection() = default;
	__device__
	BlinnPhongReflection(const vec3& Ks,
						 const float& phong_exponent, const vec3& n) :
		Ks(Ks), pn(phong_exponent), n(n)
	{}
	__device__
		virtual ~BlinnPhongReflection() {}
	__device__
		virtual vec3 f(const vec3& wi, const vec3& wo) const override
	{
		vec3 h = normalize(wi + wo);
		float cos_alpha = fmaxf(0.0f, dot(n, h));  // =cos(alpha) where alpha is the angle between half vector and normal                         
		float norm_factor = ((pn + 2.0f) * (pn + 4.0f)) / ((8.0f * pi()) * (powf(2, -pn / 2.0f) + pn));

		return Ks * norm_factor * powf(cos_alpha, pn);
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
	vec3 f(const vec3& wi, const vec3& wo) const;
};

#endif
#if !defined(__SHADE_KERNEL_CUH__)
#define __SHADE_KERNEL_CUH__

#include "shapes/sphere.cuh"
#include "core/surfaceInteraction.cuh"

__device__
inline vec3 shade(const SurfaceInteraction& re, const vec3& p, const vec3& wo, const vec3& wi)
{
	const float invPi = 0.31830988618379067154;

	vec3 n = re.hitobject.compute_normal_at(p);
	if (dot(n, wo) < 0.0f) n = -n;
	vec3 I = vec3(1.0f, 1.0f, 1.0f);

	vec3 k_d = re.hitobject.color();

	vec3 I_a = vec3(0.01f, 0.01f, 0.01f);
	vec3 k_a = k_d;

	vec3 h = normalize(wo + wi);
	//vec3 r = (2.0f * dot(n, wi) * n) - wi;
	vec3 k_s = re.hitobject.specular_color();
	float shininess = re.hitobject.shininess();

	vec3 L = //k_a * I_a +
		(k_d * invPi) * I * fmaxf(0.0f, dot(n, wi));
		//k_s * I * powf(fmaxf(0.0f, dot(n, h)), shininess);
	return L;
}

#endif
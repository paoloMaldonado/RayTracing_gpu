#if !defined(__SHADE_KERNEL_CUH__)
#define __SHADE_KERNEL_CUH__

#include "shapes/sphere.cuh"

__device__
inline vec3 shade(const vec3& ipoint, const Sphere& iobject, const vec3& view_dir, const vec3& light_source_pos)
{
	vec3 n = iobject.compute_normal_at(ipoint);
	if (dot(n, view_dir) < 0.0f) n = -n;
	vec3 l = normalize(light_source_pos - ipoint);
	vec3 I = vec3(1.0f, 1.0f, 1.0f);

	vec3 k_d = iobject.color();

	vec3 I_a = vec3(0.1f, 0.1f, 0.1f);
	vec3 k_a = k_d;

	vec3 h = normalize(view_dir + l);
	vec3 r = (2.0f * dot(n, l) * n) - l;
	vec3 k_s = iobject.specular_color();
	float shininess = iobject.shininess();

	vec3 L = k_a * I_a +
			 k_d * I * fmaxf(0.0f, dot(n, l)) +
			 k_s * I * powf(fmaxf(0.0f, dot(n, h)), shininess);
	return L;
}

#endif
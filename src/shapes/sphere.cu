#include "sphere.cuh"

__device__ 
bool Sphere::hitted_by(const Ray& ray, float& t) const
{
	// discriminant = B^-4AC
	vec3 oc = ray.origin - center;
	float A = dot(ray.direction, ray.direction);
	float half_B = dot(ray.direction, oc);
	float C = dot(oc, oc) - (radius * radius);

	float discriminant = half_B*half_B - A*C;
	if (discriminant >= 0.0f)
	{
		float t0 = (-half_B - sqrtf(discriminant)) / A;
		float t1 = (-half_B + sqrtf(discriminant)) / A;

		if (t0 > 0.0f)
		{
			t = t0;
			return true;
		}

		if (t1 > 0.0f)
		{
			t = t1;
			return true;
		}
	}
	return false;
}

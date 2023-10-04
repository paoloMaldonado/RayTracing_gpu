#include "compound.cuh"
#include "math.h"

__device__
bool Compound::hitted_by(const Ray& ray, float& t) const
{
	float inf = INFINITY;
	float tmin = inf;
	bool hit = false;

	for (int i = 0; i < nObjects; ++i)
	{
		if (objects[i]->hitted_by(ray, t) && (t < tmin))
		{
			hit = true;
			tmin = t;
			isect_object = objects[i];
		}
	}

	if (hit)
		t = tmin;

	return hit;
}

__device__
normal3 Compound::compute_normal_at(const point3& p) const
{
	return isect_object->compute_normal_at(p);
}

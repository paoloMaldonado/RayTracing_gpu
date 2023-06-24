#if !defined(__INTERSECTION_KERNEL_CUH__)
#define __INTERSECTION_KERNEL_CUH__

#include "misc/ray.cuh"
#include "shapes/sphere.cuh"
#include "core/surfaceInteraction.cuh"


__device__
inline bool intersection(const Ray& ray, const Sphere* object_list, const unsigned int N, SurfaceInteraction& rec)
{
	bool hit = false;
	float t_0;
	SurfaceInteraction temp_rec;
	temp_rec.t = INFINITY;

	for (unsigned int i = 0; i < N; i++)
	{
		if (object_list[i].hitted_by(ray, t_0))
		{
			if (t_0 < temp_rec.t)
			{
				hit = true;
				temp_rec.t = t_0;
				temp_rec.hitobject = object_list[i];
				rec = temp_rec;
			}
		}
	}
	return hit;
}

#endif
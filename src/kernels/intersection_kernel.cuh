#if !defined(__INTERSECTION_KERNEL_CUH__)
#define __INTERSECTION_KERNEL_CUH__

#include "core/ray.cuh"
#include "core/surfaceInteraction.cuh"

__device__
inline bool intersection(const Ray& ray, Instance** object_list, const unsigned int N, SurfaceInteraction& rec)
{
	bool hit = false;
	float t_0;

	for (unsigned int i = 0; i < N; i++)
	{
		Ray inv_ray;
		if (object_list[i]->hitted_by(ray, t_0, inv_ray))
		{
			if (t_0 < rec.t)
			{
				hit = true;
				
				// minimun intersection distance
				rec.t         = t_0;
				// compute intersection point
				rec.p         = ray.point_at_parameter(t_0);
				// compute wo (outgoing) direction
				rec.wo        = -ray.direction;
				// intersected primitive
				rec.hitobject = object_list[i]->object_ptr;   // hitobject will live as long as shape lives (both holds the same address)
				// normal at intersection point (transformed normal)
				point3 p_at_untransformed = inv_ray.point_at_parameter(t_0);
				rec.n         = object_list[i]->compute_normal(p_at_untransformed);
			}
		}
	}
	return hit;
}

__device__
inline bool intersectionShadow(const Ray& ray, Instance** object_list, const unsigned int N)
{
	float t_0;

	for (unsigned int i = 0; i < N; i++)
	{
		Ray inv_ray;
		if (object_list[i]->hitted_by(ray, t_0, inv_ray))
		{
			return true;
		}
	}
	return false;
}

#endif
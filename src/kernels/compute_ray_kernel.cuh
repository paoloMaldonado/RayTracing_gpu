#if !defined(__COMPUTE_RAY_KERNEL_CUH__)
#define __COMPUTE_RAY_KERNEL_CUH__

#include "misc/vec3.cuh"
#include "misc/camera.cuh"
#include "misc/ray.cuh"

__device__
inline Ray compute_ray(const int& i, const int& j, Camera cam, const float& r, const float& t, const int WIDTH, const int HEIGHT)
{
	float l = -r;
	float b = -t;

	// where (r-l)/WIDTH , (t-b)/HEIGHT are the pixel sizes for row and columns respectively [Shirley] 
	float u_prime = l + (r - l) * (i + 0.5f) / WIDTH;
	float v_prime = b + (t - b) * (j + 0.5f) / HEIGHT;
	// another approach by [Suffern] where S = pixel size and can be specified by the user
	//float u_prime = (r - l) / WIDTH * (i - WIDTH / 2.0f + 0.5f);
	//float v_prime = (t - b) / HEIGHT * (j - HEIGHT / 2.0f + 0.5f);

	Ray ray;
	ray.origin = cam.e;
	ray.direction = normalize(-cam.d * cam.w + u_prime * cam.u + v_prime * cam.v);

	return ray;
}

#endif
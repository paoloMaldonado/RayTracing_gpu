#if !defined(__CUDA_RAY_TRACING_CUH__)
#define __CUDA_RAY_TRACING_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels/compute_ray_kernel.cuh"
#include "kernels/intersection_kernel.cuh"
#include "kernels/shade_kernel.cuh"

class PointLight;

void callRayTracingKernel(
    float4* d_pixel,
    Instance** object_list,
    unsigned int N,
    Camera camera,
    point3 point_light,
    const int width,
    const int height);

__device__
Spectrum specularBounces(const SurfaceInteraction& isect, const int maxDepth, Instance** scene, const int& N, PointLight light, MemoryManager& memory);
__device__
Spectrum specularReflect(Ray& ray, SurfaceInteraction& isect, Instance** scene, const int& N, PointLight light, MemoryManager& memory);
__device__
Spectrum specularRefract(Ray& ray, SurfaceInteraction& isect, Instance** scene, const int& N, PointLight light, MemoryManager& memory);

#endif



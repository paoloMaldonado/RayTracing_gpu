#if !defined(__CUDA_RAY_TRACING_CUH__)
#define __CUDA_RAY_TRACING_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "kernels/compute_ray_kernel.cuh"
#include "kernels/intersection_kernel.cuh"
#include "kernels/shade_kernel.cuh"

void callRayTracingKernel(
    dim3 grid,
    dim3 thread_block,
    float4* d_pixel,
    Sphere* object_list,
    unsigned int N,
    Camera camera,
    vec3 point_light,
    const int width,
    const int height);

#endif



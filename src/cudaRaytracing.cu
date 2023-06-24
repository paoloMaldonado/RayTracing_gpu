#include "cudaRaytracing.cuh"

__global__
void render(float4* pixel, Sphere* object_list, unsigned int N, Camera camera, vec3 point_light, const int width, const int height)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int offset = x + y * blockDim.x * gridDim.x;

    //const float aspect_ratio = 16.0 / 9.0;
    const float aspect_ratio = static_cast<float>(width) / height;
    float t = 1.0f;
    float r = aspect_ratio * t;

    SurfaceInteraction rec;
    Ray ray = compute_ray(x, y, camera, r, t, width, height);
    bool hit = intersection(ray, object_list, N, rec);

    //pixel[offset] = make_float4(0.54f, 0.54f, 0.54f, 1.0f);
    pixel[offset] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    if (hit) // there is an intersection
    {
        vec3 p = ray.point_at_parameter(rec.t);
        vec3 wi = normalize(point_light - p);
        vec3 wo = -ray.direction;
        vec3 color = shade(rec, p, wo, wi);
        pixel[offset] = make_float4(color.x, color.y, color.z, 1.0f);
    }    

}

void callRayTracingKernel(
    float4* d_pixel,
    Sphere* object_list,
    unsigned int N,
    Camera camera,
    vec3 point_light,
    const int width,
    const int height)
{
    dim3 thread_block(8, 8, 1);
    dim3 grid(width / thread_block.x, height / thread_block.y, 1);
    render<<< grid, thread_block >>>(d_pixel, object_list, N, camera, point_light, width, height);
}
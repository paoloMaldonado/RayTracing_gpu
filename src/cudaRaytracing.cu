#include "cudaRaytracing.cuh"

__global__
void render(float4* pixel, Sphere* object_list, unsigned int N, Camera camera, vec3 point_light, const int width, const int height)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int offset = x + y * blockDim.x * gridDim.x;

    const float aspect_ratio = 16.0 / 9.0;
    float t = 20.0f;
    float r = aspect_ratio * t;

    hit_record rec;
    Ray ray = compute_ray(x, y, camera, r, t, width, height);
    bool hit = intersection(ray, object_list, N, rec);

    pixel[offset] = make_float4(0.54f, 0.54f, 0.54f, 1.0f);

    if (hit) // there is an intersection
    {
        vec3 color = shade(ray.point_at_parameter(rec.t), rec.hitobject, -ray.direction, point_light);
        pixel[offset] = make_float4(color.x, color.y, color.z, 1.0f);
    }    

}

void callRayTracingKernel(
    dim3 grid,
    dim3 thread_block,
    float4* d_pixel,
    Sphere* object_list,
    unsigned int N,
    Camera camera,
    vec3 point_light,
    const int width,
    const int height)
{
    render<<< grid, thread_block >>>(d_pixel, object_list, N, camera, point_light, width, height);
}
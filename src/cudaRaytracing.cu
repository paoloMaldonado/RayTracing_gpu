#include "cudaRaytracing.cuh"
#include "core/visibilityTester.cuh"
#include "lights/pointLight.cuh"

__global__
void render(float4* pixel, Shape** object_list, unsigned int N, Camera camera, vec3 point_light, const int width, const int height)
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

    if (hit) // if there is an intersection
    {
        vec3 wi;
        bool in_shadow;
        VisibilityTester visibility(object_list, N);

        PointLight light(point_light);
        vec3 I = light.sample_li(rec, ray, visibility, wi, in_shadow);

        //vec3 I = point_light.sample_li(ray, rec, )
        //vec3 p = ray.point_at_parameter(rec.t);
        //vec3 wi = normalize(point_light - p);
        
        //bool visibility = 0;
        //Ray shadow_ray(p + wi*0.0001f, wi);
        
        //visibility = intersectionShadow(shadow_ray, object_list, N);

        // preallocate a buffer for placement new -> faster than dynamic allocation 
        MemoryManager memory;
        vec3 color = shade(rec, wi, memory, in_shadow);

        pixel[offset] = make_float4(color.x, color.y, color.z, 1.0f);
    }    
}

void callRayTracingKernel(
    float4* d_pixel,
    Shape** object_list,
    unsigned int N,
    Camera camera,
    vec3 point_light,
    const int width,
    const int height)
{
    dim3 thread_block(8, 8, 1);
    dim3 grid(width / thread_block.x, height / thread_block.y, 1);
    render<<< grid, thread_block >>>(d_pixel, object_list, N, camera, point_light, width, height);
    cudaDeviceSynchronize();
}
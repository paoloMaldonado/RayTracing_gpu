#include "cudaRaytracing.cuh"
#include "core/visibilityTester.cuh"
#include "lights/pointLight.cuh"
#include "core/dstructs.cuh"

#include <iostream>

__global__
void render(float4* pixel, Instance** object_list, unsigned int N, Camera camera, point3 point_light, const int width, const int height)
{
    int maxDepth = 3;

    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int offset = x + y * blockDim.x * gridDim.x;

    //const float aspect_ratio = 16.0 / 9.0;
    const float aspect_ratio = static_cast<float>(width) / height;
    float t = 1.0f;
    float r = aspect_ratio * t;

    Ray ray = compute_ray(x, y, camera, r, t, width, height);

    SurfaceInteraction rec;
    bool hit = intersection(ray, object_list, N, rec);

    pixel[offset] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    if (hit) // if there is an intersection
    {
        // preallocate a buffer for placement new -> faster than dynamic allocation 
        MemoryManager memory;
        rec.compute_scattering_functions(memory);

        vec3 wi;
        bool in_shadow;
        VisibilityTester visibility(object_list, N);

        PointLight light(point_light);
        Spectrum I = light.sample_li(rec, visibility, wi, in_shadow);

        Spectrum color = shade(rec, wi, in_shadow);   // for more than 1 ls -> color += shade() and inside for loop

        // trace rays for specular reflection and refraction
        ray.direction = rec.wo;
        //for (int i = 0; i < maxDepth - 1; ++i)
        //{
        //    //color += specularReflect(ray, rec, object_list, N, light, memory);
        //    color += specularRefract(ray, rec, object_list, N, light, memory);
        //}
        color += specularBounces(rec, maxDepth, object_list, N, light, memory);

        pixel[offset] = make_float4(color.x, color.y, color.z, 1.0f);
    }    
}

void callRayTracingKernel(
    float4* d_pixel,
    Instance** object_list,
    unsigned int N,
    Camera camera,
    point3 point_light,
    const int width,
    const int height)
{
    dim3 thread_block(8, 8, 1);
    dim3 grid(width / thread_block.x, height / thread_block.y, 1);
    render<<< grid, thread_block >>>(d_pixel, object_list, N, camera, point_light, width, height);
    cudaDeviceSynchronize();
}

__device__
Spectrum specularBounces(const SurfaceInteraction& isect, const int maxDepth, Instance** scene, const int& N, PointLight light, MemoryManager& memory)
{
    const int refraction_rays = (powf(2, maxDepth) - 2) / 2;

    Spectrum color(0.0f);
    Stack<Ray> refraction_stack(refraction_rays);
    Stack<int> depth_stack(maxDepth);
    Stack<Spectrum> fresnel_stack(refraction_rays);
    Stack<normal3> normal_stack(refraction_rays);

    int tree_depth = 1;
    bool continue_loop = true;

    Ray ray;

    vec3 wo = isect.wo;         // outgoing direction
    vec3 wr;                    // reflected/transmited direction
    vec3 wt;                    // transmited direction
    normal3 n = isect.n;        // normal at intersection point
    vec3 wi;                    // incident direction (either reflected/transmited) -- to be used in the loop

    // Specular reflection
    Spectrum f = isect.bsdf.sample_f(wo, wr, BxDFType::SPECULAR_REFLECTION);
    if (!f.isBlack() && fabs(dot(wr, n)) != 0.0f && tree_depth < maxDepth)
    {
        ray = Ray(isect.p + wr * 0.0001f, wr);

        // if there Specular refraction
        Spectrum f_t = isect.bsdf.sample_f(wo, wt, BxDFType::SPECULAR_REFRACTION);
        if (!f_t.isBlack() && fabs(dot(wt, n)) != 0.0f && tree_depth < maxDepth)
        {
            refraction_stack.push(Ray(isect.p + wt * 0.0001f, wt));
            depth_stack.push(tree_depth);
            fresnel_stack.push(f_t);
            normal_stack.push(n);
        }
    }

    while (continue_loop)
    {
        wi = ray.direction;
        if (!f.isBlack() && fabs(dot(wi, n)) != 0.0f && tree_depth < maxDepth)
        {
            SurfaceInteraction isect;
            bool hit = intersection(ray, scene, N, isect);

            if (hit) // if there is an intersection
            {
                tree_depth += 1;

                isect.n = dot(isect.n, ray.direction) < 0.0f ? -isect.n : isect.n;
                isect.compute_scattering_functions(memory);

                vec3 wi;  // light direction coming from the light source
                bool in_shadow;
                VisibilityTester visibility(scene, N);

                Spectrum I = light.sample_li(isect, visibility, wi, in_shadow);

                color += f * shade(isect, wi, in_shadow) * fabs(dot(ray.direction, n));

                // ------------------------------------------------------------------
                vec3 wo = isect.wo;         // outgoing direction
                vec3 wr;                    // reflected/transmited direction
                vec3 wt;                    // transmited direction
                n = isect.n;                // normal at intersection point

                // Specular reflection
                Spectrum f = isect.bsdf.sample_f(wo, wr, BxDFType::SPECULAR_REFLECTION);
                if (!f.isBlack() && fabs(dot(wr, n)) != 0.0f)
                {
                    ray = Ray(isect.p + wr * 0.0001f, wr);

                    // Specular refraction
                    Spectrum f_t = isect.bsdf.sample_f(wo, wt, BxDFType::SPECULAR_REFRACTION);
                    if (!f_t.isBlack() && fabs(dot(wt, n)) != 0.0f)
                    {
                        refraction_stack.push(Ray(isect.p + wt * 0.0001f, wt));
                        depth_stack.push(tree_depth);
                        fresnel_stack.push(f_t);
                        normal_stack.push(n);
                    }
                }
                else
                    continue_loop = false;
                // -------------------------------------------------------------------
            }
            else
                continue_loop = false;
        }
        else
        {
            color += Spectrum(0.0f);
            continue_loop = false;
        }

        if (!continue_loop && !refraction_stack.isEmpty() && 
            !depth_stack.isEmpty() && 
            !fresnel_stack.isEmpty() && 
            !normal_stack.isEmpty())
        {
            ray = refraction_stack.pop();
            tree_depth = depth_stack.pop();
            f = fresnel_stack.pop();
            n = normal_stack.pop();
            continue_loop = true;
        }
    }
    return color;
}

__device__
Spectrum specularReflect(Ray& ray, SurfaceInteraction& isect, Instance** scene, const int& N, PointLight light, MemoryManager& memory)
{
    vec3 wo = isect.wo;
    vec3 wi;  // specular reflected direction

    normal3 n = isect.n;
    BxDFType type = BxDFType::SPECULAR_REFLECTION;
    Spectrum f = isect.bsdf.sample_f(wo, wi, type);
    
    if (!f.isBlack() && fabs(dot(wi, n)) != 0.0f)
    {
        ray = Ray(isect.p + wi * 0.0001f, wi);

        SurfaceInteraction t_isect;
        bool hit = intersection(ray, scene, N, t_isect);

        // clear bsdf array before adding new BxDFs
        isect.bsdf.clear();
        isect = t_isect;

        if (hit) // if there is an intersection
        {
            isect.n = dot(isect.n, ray.direction) < 0.0f ? -isect.n : isect.n;
            isect.compute_scattering_functions(memory);

            vec3 wi;  // light direction coming from the light source
            bool in_shadow;
            VisibilityTester visibility(scene, N);

            Spectrum I = light.sample_li(isect, visibility, wi, in_shadow);

            Spectrum color = f * shade(isect, wi, in_shadow) * fabs(dot(ray.direction, n));
            return color;
        }
        else
            return Spectrum(0.0f);
    }
    else
        return Spectrum(0.0f);

}

__device__
Spectrum specularRefract(Ray& ray, SurfaceInteraction& isect, Instance** scene, const int& N, PointLight light, MemoryManager& memory)
{
    vec3 wo = isect.wo;
    vec3 wi;            // specular refracted direction

    normal3 n = isect.n;   
    BxDFType type = BxDFType::SPECULAR_REFRACTION;
    Spectrum f = isect.bsdf.sample_f(wo, wi, type);
    
    if (!f.isBlack() && fabs(dot(wi, n)) != 0.0f)
    {
        ray = Ray(isect.p + wi * 0.0001f, wi);
        
        SurfaceInteraction t_isect;
        bool hit = intersection(ray, scene, N, t_isect);

        // clear bsdf array before adding new BxDFs
        isect.bsdf.clear();
        isect = t_isect;

        if (hit) // if there is an intersection
        {
            // ensure the normal is always pointing outside of the object
            isect.n = dot(isect.n, ray.direction) < 0.0f ? -isect.n : isect.n;

            isect.compute_scattering_functions(memory);

            vec3 wi;  // light direction coming from the light source
            bool in_shadow;
            VisibilityTester visibility(scene, N);

            Spectrum I = light.sample_li(isect, visibility, wi, in_shadow);

            Spectrum color = f * shade(isect, wi, in_shadow) * fabs(dot(ray.direction, n));
            return color;

        }
        else
            return Spectrum(0.0f);

    }
    else
        return Spectrum(0.0f);
}
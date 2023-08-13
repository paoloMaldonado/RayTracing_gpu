#include "Scene.cuh"
#include "shapes/sphere.cuh"
#include "materials/matte.cuh"
#include "materials/plastic.cuh"
#include "materials/mirror.cuh"
#include "materials/glass.cuh"

__global__
void create_scene(Shape** objects, const unsigned int N, Material** materials, const unsigned int nMat)
{
    //Colors
    vec3 RED = vec3(1.0f, 0.0f, 0.0f);
    vec3 GREEN = vec3(0.0f, 1.0f, 0.0f);
    vec3 BLUE = vec3(0.0f, 0.0f, 1.0f);
    vec3 ORANGE = vec3(1.0f, 0.4f, 0.02f);
    vec3 BLACK = vec3(0.0f, 0.0f, 0.0f);

    *(materials)     = new MatteMaterial(ORANGE);
    *(materials + 1) = new MatteMaterial(GREEN);
    //*(materials + 2) = new PlasticMaterial(vec3(0.0f), vec3(1.0f), 1000.0f, false);
    *(materials + 2) = new GlassMaterial(vec3(1.0f), vec3(1.0f), 1.5f);

    *(objects)     = new Sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, *(materials));
    *(objects + 1) = new Sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f, *(materials + 1));
    *(objects + 2) = new Sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f, *(materials + 2));
    *(objects + 3) = new Sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f, *(materials + 1));
}

__global__
void destroy_scene(Shape** objects, Material** materials)
{
    delete *(objects);
    delete *(objects + 1);
    delete *(objects + 2);
    delete *(objects + 3);

    delete *(materials);
    delete *(materials + 1);
    delete *(materials + 2);

}

Scene::Scene(Shape** list_objects, const unsigned int& N, Material** list_materials, const unsigned int& nMat) 
    : d_objects(nullptr), primitive_count(N), d_materials(list_materials), material_count(nMat)
{}

void Scene::build()
{
	unsigned int nObjects = primitive_count * sizeof(Shape*);
	cudaMalloc(&d_objects, nObjects);
    unsigned int nMat = material_count * sizeof(Material*);
    cudaMalloc(&d_materials, nMat);
	create_scene<<<1, 1>>>(d_objects, primitive_count, d_materials, material_count);
    cudaDeviceSynchronize();
}

void Scene::destroy()
{
    destroy_scene<<<1, 1>>>(d_objects, d_materials);
    cudaDeviceSynchronize();
	cudaFree(d_objects);
}


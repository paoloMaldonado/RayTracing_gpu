#include "scene.cuh"
#include "shapes/sphere.cuh"
#include "shapes/instance.cuh"
#include "materials/matte.cuh"
#include "materials/plastic.cuh"
#include "materials/mirror.cuh"
#include "materials/glass.cuh"

__global__
void create_scene(Shape** objects, Instance** instances, Material** materials)
{
    //Colors
    Spectrum RED = Spectrum(1.0f, 0.0f, 0.0f);
    Spectrum GREEN = Spectrum(0.0f, 1.0f, 0.0f);
    Spectrum BLUE = Spectrum(0.0f, 0.0f, 1.0f);
    Spectrum ORANGE = Spectrum(1.0f, 0.4f, 0.02f);
    Spectrum BLACK = Spectrum(0.0f, 0.0f, 0.0f);

    *(materials)     = new MatteMaterial(BLUE);
    *(materials + 1) = new MatteMaterial(GREEN);
    //*(materials + 2) = new PlasticMaterial(ORANGE, Spectrum(1.0f), 1000.0f, false);
    *(materials + 2) = new GlassMaterial(Spectrum(1.0f), Spectrum(1.0f), 1.5f);

    *(objects)     = new Sphere(point3(0.0f, 0.0f, 0.0f), 1.0f, materials[0]);
    *(objects + 1) = new Sphere(point3(0.0f, 0.0f, 0.0f), 1.0f, *(materials + 1));
    *(objects + 2) = new Sphere(point3(0.0f, 0.0f, 0.0f), 1.0f, *(materials + 2));
    *(objects + 3) = new Sphere(point3(0.0f, 0.0f, 0.0f), 1.0f, *(materials + 1));

    // read from left to right because they are inverse matrices
    Transform center  = scaling(0.5f, 0.5f, 0.5f) * translate(vec3(0.0f, 0.0f, -1.0f));
    Transform left    = scaling(0.5f, 0.5f, 0.5f) * translate(vec3(-1.0f, 0.0f, -1.0f));
    Transform right   = scaling(0.5f, 0.5f, 0.5f) * translate(vec3(1.0f, 0.0f, -1.0f));
    Transform floor   = scaling(100.0f, 100.0f, 100.0f) * translate(vec3(0.0f, -100.5f, -1.0f));

    *(instances)      = new Instance(objects[0], center);
    *(instances + 1)  = new Instance(objects[1], left);
    *(instances + 2)  = new Instance(objects[2], right);
    *(instances + 3)  = new Instance(objects[3], floor);
}

__global__
void destroy_scene(Shape** objects, Instance** instances, Material** materials)
{
    delete *(objects);
    delete *(objects + 1);
    delete *(objects + 2);
    delete *(objects + 3);

    delete *(instances);
    delete *(instances + 1);
    delete *(instances + 2);
    delete *(instances + 3);

    delete *(materials);
    delete *(materials + 1);
    delete *(materials + 2);

}

Scene::Scene(Shape** list_shapes, Instance** list_instances, const unsigned int& N_obj, const unsigned int& N, Material** list_materials, const unsigned int& nMat)
    : d_shapes(nullptr), d_instances(nullptr), primitive_count(N_obj), 
      instance_count(N), d_materials(list_materials), material_count(nMat)
{}

void Scene::build()
{
	unsigned int nObjects = primitive_count * sizeof(Shape*);
	cudaMalloc(&d_shapes, nObjects);
    unsigned int nInstances = instance_count * sizeof(Instance*);
    cudaMalloc(&d_instances, nInstances);
    unsigned int nMat = material_count * sizeof(Material*);
    cudaMalloc(&d_materials, nMat);
	create_scene<<<1, 1>>>(d_shapes, d_instances, d_materials);
    cudaDeviceSynchronize();
}

void Scene::destroy()
{
    destroy_scene<<<1, 1>>>(d_shapes, d_instances, d_materials);
    cudaDeviceSynchronize();
	cudaFree(d_shapes);
    cudaFree(d_instances);
    cudaFree(d_materials);
}


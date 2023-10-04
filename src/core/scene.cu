#include "scene.cuh"
#include "shapes/sphere.cuh"
#include "shapes/triangle.cuh"
#include "shapes/instance.cuh"
#include "materials/matte.cuh"
#include "materials/plastic.cuh"
#include "materials/mirror.cuh"
#include "materials/glass.cuh"

__global__
void makeShapes(Shape** shapes, const unsigned int nPrimitives)
{
    //Shape* sphere = createSphereShape(point3(0.0f, 0.0f, 0.0f), 1.0f);
    //shapes[0] = sphere;

    //sphere = createSphereShape(point3(0.0f, 0.0f, 0.0f), 1.0f);
    //shapes[1] = sphere;

    int indices[9] = { 0, 1, 2, 1, 3, 2, 0, 3, 1 };
    point3 vertices[4] = { point3(-1.0f, -0.5f, 0.0f),
                           point3(0.5f, 0.5f, 0.0f),
                           point3(0.5f, 1.5f, 0.0f),
                           point3(1.0f, 0.0f, 0.0f) };

    Shape** s = createTriangleMeshShape(4, 3, indices, vertices);

    for (int i = 0; i < nPrimitives; ++i)
    {
        shapes[i] = s[i];
    }

    delete[] s;
}


__global__
void makeMaterials(Material** materials, unsigned int nMaterials)
{
    //Colors
    Spectrum RED = Spectrum(1.0f, 0.0f, 0.0f);
    Spectrum GREEN = Spectrum(0.0f, 1.0f, 0.0f);
    Spectrum BLUE = Spectrum(0.0f, 0.0f, 1.0f);
    Spectrum ORANGE = Spectrum(1.0f, 0.4f, 0.02f);
    Spectrum BLACK = Spectrum(0.0f, 0.0f, 0.0f);

    materials[0] = new MatteMaterial(BLUE);
    materials[1] = new MatteMaterial(GREEN);
    //materials[2] = new PlasticMaterial(ORANGE, Spectrum(1.0f), 1000.0f, false);
    materials[2] = new GlassMaterial(Spectrum(1.0f), Spectrum(1.0f), 1.5f);
}

__global__
void create_scene(Shape** shapes, Instance** instances, Material** materials)
{
    // read from left to right because these are inverse matrices
    Transform center  = scaling(0.5f, 0.5f, 0.5f) * translate(vec3(0.0f, 0.0f, -1.0f));
    //Transform left    = scaling(0.5f, 0.5f, 0.5f) * translate(vec3(-1.0f, 0.0f, -1.0f));
    //Transform right   = scaling(0.5f, 0.5f, 0.5f) * translate(vec3(1.0f, 0.0f, -1.0f));
    //Transform floor   = scaling(100.0f, 100.0f, 100.0f) * translate(vec3(0.0f, -100.5f, -1.0f));

    instances[0] = new Instance(shapes[0], materials[0]);
    instances[1] = new Instance(shapes[1], materials[0]);
    instances[2] = new Instance(shapes[2], materials[0]);
}

__global__
void destroy_scene(Shape** objects, Instance** instances, 
    Material** materials, unsigned int nPrimitives, unsigned int nMaterials)
{
    // Deleting primitives
    for (int i = 0; i < nPrimitives; ++i)
    {
        delete objects[i];
    }

    // Deleting instances
    for (int i = 0; i < nPrimitives; ++i)
    {
        delete instances[i];
    }

    // Deleting materials
    for (int i = 0; i < nMaterials; ++i)
    {
        delete materials[i];
    }
}

Scene::Scene(const unsigned int& nPrimitives, const unsigned int& nMaterials)
    : nPrimitives(nPrimitives), nMaterials(nMaterials)
{}

void Scene::build()
{
	cudaMalloc(&d_shapes, nPrimitives * sizeof(Shape*));
    cudaMalloc(&d_instances, nPrimitives * sizeof(Instance*));
    cudaMalloc(&d_materials, nMaterials * sizeof(Material*));

    makeMaterials<<<1, 1>>>(d_materials, nMaterials);
    cudaDeviceSynchronize();

    // Create array of shapes and materials objects
    makeShapes<<<1, 1>>>(d_shapes, nPrimitives);
    cudaDeviceSynchronize();
    
	create_scene<<<1, 1>>>(d_shapes, d_instances, d_materials);
    cudaDeviceSynchronize();
}

void Scene::destroy()
{
    destroy_scene<<<1, 1>>>(d_shapes, d_instances, d_materials, nPrimitives, nMaterials);
    cudaDeviceSynchronize();
    cudaFree(d_shapes);
    cudaFree(d_instances);
    cudaFree(d_materials);
}


#include "scene.cuh"
#include "shapes/sphere.cuh"
#include "shapes/triangle.cuh"
#include "shapes/instance.cuh"
#include "materials/matte.cuh"
#include "materials/plastic.cuh"
#include "materials/mirror.cuh"
#include "materials/glass.cuh"

#include "utility/utils.cuh"
#include <iostream>

__global__
void showOBJ(int* indices, point3* vertices, MatParam* material_params, int* offset_per_shape,
    unsigned int size_indices, unsigned int size_vertices, unsigned int n_triangles_total)
{
    //for (int i = 0; i < size_vertices; i++)
    //{
    //    printf("%f ", vertices[i].x);
    //    printf("%f ", vertices[i].y);
    //    printf("%f ", vertices[i].z);
    //    printf("\n");
    //}

    printf("%f ", material_params[0].Kd[0]);
    printf("%f ", material_params[0].Kd[1]);
    printf("%f ", material_params[0].Kd[2]);
    printf("\n");
}


__device__
class Instance_stack
{
public:
    __device__
    Instance_stack() : pos(0), last_pos(0)
    {}
    __device__
    void push_back(Instance** instances, Shape** prims, Transform* transform, Material* material, const int nPrimitives)
    {
        int i = 0;
        for (pos; pos < last_pos + nPrimitives; ++pos)
        {
            instances[pos] = new Instance(prims[i++], transform, material);
        }

        delete[] prims;
        last_pos = pos;
    }

private:
    mutable size_t pos;
    mutable size_t last_pos;
};


__device__
Shape** makeShapes(const char* name, const unsigned int nPrimitives, int* indices, point3* vertices)
{
    Shape** prims = new Shape*[nPrimitives];
    Shape* s = nullptr;

    if(cuda_strcmp(name, "sphere") == 0)
        s = createSphereShape(point3(0.0f, 0.0f, 0.0f), 1.0f);

    if (s != nullptr)
    {
        for (int i = 0; i < nPrimitives; ++i)
        {
            prims[i] = s;
        }
    }

    else if (cuda_strcmp(name, "trianglemesh") == 0)
    {
        //int indices[36] = { 0, 6, 4, 0, 2, 6, 0, 3, 2, 0, 1, 3, 2, 7, 6, 2, 3, 7, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 1, 5, 7, 1, 7, 3 };
        //point3 vertices[8] = { point3(0.0f,  0.0f,  0.0f),
        //                       point3(0.0f,  0.0f,  1.0f),
        //                       point3(0.0f,  1.0f,  0.0f),
        //                       point3(0.0f,  1.0f,  1.0f),
        //                       point3(1.0f,  0.0f,  0.0f),
        //                       point3(1.0f,  0.0f,  1.0f),
        //                       point3(1.0f,  1.0f,  0.0f),
        //                       point3(1.0f,  1.0f,  1.0f) };

        Shape** multi_s = createTriangleMeshShape(nPrimitives, indices, vertices);

        for (int i = 0; i < nPrimitives; ++i)
        {
            prims[i] = multi_s[i];
        }
        delete[] multi_s;
    }

    return prims;
}


__device__
Material* makeMaterial(const MatParam& material_parameters)
{
    if (material_parameters.ilum == MaterialType::MATTE)
    {
        Spectrum Kd = material_parameters.Kd;
        return new MatteMaterial(Kd);
    }
}

__global__
void makeTransforms(Transform** transforms, unsigned int nTransforms)
{
    // read from left to right because these are inverse matrices
    Transform* center = new Transform(scaling(0.5f, 0.5f, 0.5f) * translate(vec3(0.0f, 0.0f, -1.0f)));
    Transform* left   = new Transform(scaling(0.5f, 0.5f, 0.5f) * translate(vec3(-1.0f, 0.0f, -1.0f)));
    Transform* right  = new Transform(scaling(0.5f, 0.5f, 0.5f) * translate(vec3(1.0f, 0.0f, -1.0f)));
    //Transform floor   = scaling(100.0f, 100.0f, 100.0f) * translate(vec3(0.0f, -100.5f, -1.0f));

    transforms[0] = center;
    transforms[1] = left;
    transforms[2] = right;
}

__global__
void create_scene(Instance** instances, Material** materials, MatParam* material_params,  Transform** transforms, 
    const int nPrimitives, int* indices = nullptr, point3* vertices = nullptr, int* triangles_per_shape = nullptr, 
    int nObjectsInMesh = 1)
{
    Instance_stack list;

    Shape** shapes = nullptr;

    size_t srcIdx = 0;
    for (size_t p = 0; p < nObjectsInMesh; ++p)
    {
        int indices_in_shape = triangles_per_shape[p] * 3;
        int* indices_offset = new int[indices_in_shape];
        memcpy(indices_offset, &indices[srcIdx], indices_in_shape * sizeof(int));

        // instantiate each shape
        shapes = makeShapes("trianglemesh", triangles_per_shape[p], indices_offset, vertices);
        // instantiate each material for the shape with index p (Assuming each shape has its own material)
        materials[p] = makeMaterial(material_params[p]);
        // pushing shapes and corresponding materials to the list and populate the device pointer 'instances' necesary for rendering
        list.push_back(instances, shapes, transforms[0], materials[p], triangles_per_shape[p]);

        srcIdx = srcIdx + indices_in_shape;  // srcIdx + triangles_per_shape[p] * 3
        delete[] indices_offset;
    }
}

__global__
void destroy_scene(Instance** instances, Material** materials, Transform** transforms, 
    unsigned int nPrimitives, unsigned int nMaterials)
{
    // Deleting primitives
    for (int i = 0; i < nPrimitives; ++i)
    {
        delete instances[i]->object_ptr;  // Deleting the previously allocated Shape (Shape*)
    }

    // Deleting materials
    for (int i = 0; i < nMaterials; ++i)
    {
        if(materials[i] != nullptr)
            delete materials[i];
    }

    // Deleting transforms
    for (int i = 0; i < 3; ++i)
    {
        delete transforms[i];
    }

    // Deleting instances
    for (int i = 0; i < nPrimitives; ++i)
    {
        delete instances[i];
    }
}

Scene::Scene(const unsigned int& nPrimitives, const unsigned int& nMaterials)
    : nPrimitives(nPrimitives), nMaterials(nMaterials)
{
    d_indices             = nullptr;
    d_vertices            = nullptr;
    d_triangles_per_shape = nullptr;
    nObjectsInMesh        = 1;
}

Scene::Scene(const unsigned int& nMaterials) : nPrimitives(0), nMaterials(nMaterials)
{
    d_indices = nullptr;
    d_vertices = nullptr;
    d_triangles_per_shape = nullptr;
    nObjectsInMesh = 1;
}

void Scene::load_obj_to_gpu(std::string inputfile, std::string materialpath)
{
    obj_loader = Loader(inputfile, materialpath);

    // Allocating indices on gpu
    int size_indices = obj_loader.Indices().size();
    cudaMalloc(&d_indices, sizeof(int) * size_indices);
    cudaMemcpy(d_indices, obj_loader.Indices().data(), sizeof(int) * size_indices, cudaMemcpyHostToDevice);

    // Allocating vertices on gpu
    int size_vertices = obj_loader.Vertices().size();
    cudaMalloc(&d_vertices, sizeof(point3) * size_vertices);
    cudaMemcpy(d_vertices, obj_loader.Vertices().data(), sizeof(point3) * size_vertices, cudaMemcpyHostToDevice);

    // Allocating faces_offset_indices on gpu
    nObjectsInMesh = obj_loader.shapes_number();
    cudaMalloc(&d_triangles_per_shape, sizeof(int) * nObjectsInMesh);
    cudaMemcpy(d_triangles_per_shape, obj_loader.N_triangles_per_shape().data(), sizeof(int) * nObjectsInMesh, cudaMemcpyHostToDevice);

    // Allocating space for material parameters on gpu
    int nMaterials = obj_loader.MaterialParams().size();
    cudaMalloc(&d_material_params, sizeof(MatParam) * nMaterials);
    cudaMemcpy(d_material_params, obj_loader.MaterialParams().data(), sizeof(MatParam) * nMaterials, cudaMemcpyHostToDevice);

    nPrimitives = obj_loader.TotalTriangles();

    showOBJ<<<1, 1>>>(d_indices, d_vertices, d_material_params, d_triangles_per_shape, size_indices, size_vertices, nObjectsInMesh);
}

void Scene::build()
{
	//cudaMalloc(&d_shapes, nPrimitives * sizeof(Shape*));
    cudaMalloc(&d_instances, nPrimitives * sizeof(Instance*));
    cudaMalloc(&d_materials, nMaterials * sizeof(Material*));
    cudaMalloc(&d_transforms, 3 * sizeof(Transform*));

    makeTransforms<<<1, 1 >>>(d_transforms, 3);
    cudaDeviceSynchronize();
    
	create_scene<<<1, 1>>>(d_instances, d_materials, d_material_params, d_transforms, 
                            nPrimitives, d_indices, d_vertices, d_triangles_per_shape, 
                            nObjectsInMesh);
    cudaDeviceSynchronize();
}

void Scene::destroy()
{
    destroy_scene<<<1, 1>>>(d_instances, d_materials, d_transforms, nPrimitives, nMaterials);
    cudaDeviceSynchronize();
    cudaFree(d_instances);
    cudaFree(d_materials);
    cudaFree(d_transforms);

    // OBJ to GPU
    cudaFree(d_indices);
    cudaFree(d_vertices);
    cudaFree(d_triangles_per_shape);
}


#include "Loader.h"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

Loader::Loader(std::string inputfile, std::string materialpath)
{
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = materialpath; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    attrib = reader.GetAttrib();
    shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();
    material_params.resize(shapes.size());  // assuming each shape in the file has only one corresponding material

    // Loading indices and vertices into the matrix
    
    // Loop over shapes
    int face_size = 0;
    for (size_t s = 0; s < shapes.size(); s++)
    {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]); // number of vertices in the face f_th

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++)
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v]; // idx has vertex/texture/normal e.g 1//2 
                indices.push_back(idx.vertex_index);
            }
            index_offset += fv;
            face_size++;
        }
        nTriangles_per_shape.push_back(face_size);
        face_size = 0;

        // selecting materials, where s is the index of the current shape in the iteration
        for (auto& mat : materials)
        {
            if (mat.name == shapes[s].name)
            {
                memcpy(material_params[s].Kd, mat.diffuse, sizeof(float) * 3);  // copying diffuse Kd array
                memcpy(material_params[s].Ks, mat.specular, sizeof(float) * 3); // copying specular Ks array
                material_params[s].shininess = mat.shininess;
                material_params[s].ior       = mat.ior;
                material_params[s].ilum      = mat.illum;
            }
        }
    }

    // Getting the vertices as a vector of point3
    size_t num_vertices = attrib.GetVertices().size();
    for (size_t v = 0; v < num_vertices; v += 3)
    {
        float x = attrib.vertices[v + 0];
        float y = attrib.vertices[v + 1];
        float z = attrib.vertices[v + 2];
        vertices.push_back(point3(x, y, z));
    }
}
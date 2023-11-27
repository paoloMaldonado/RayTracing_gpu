#pragma once

#include <string>
#include "tiny_obj_loader.h"
#include <vector>
#include <numeric>
#include "core/geometry.cuh"

struct MatParam
{
	float Kd[3];
	float Ks[3];
	float shininess;
	float ior;
	int ilum;
};

class Loader
{
public:
	Loader() = default;
	Loader(std::string inputfile, std::string materialpath);
	inline int shapes_number() const { return shapes.size(); }
	inline std::vector<int> Indices() const { return indices; }
	inline std::vector<int> Indices_normal() const { return indices_normal; }
	inline std::vector<point3> Vertices() const { return vertices; }
	inline std::vector<normal3> Normals() const { return normals; }
	inline std::vector<MatParam> MaterialParams() const { return material_params; }
	inline std::vector<int> N_triangles_per_shape() const { return nTriangles_per_shape; }
	inline unsigned int TotalTriangles() const { return std::accumulate(nTriangles_per_shape.begin(), nTriangles_per_shape.end(), 0); }

private:
	std::vector<int> indices;
	std::vector<int> indices_normal;
	std::vector<point3> vertices;
	std::vector<normal3> normals;
	std::vector<int> nTriangles_per_shape;
	unsigned int totalTriangles;
	std::vector<MatParam> material_params;

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
};

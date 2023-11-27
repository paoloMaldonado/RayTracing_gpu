#if !defined(__TRIANGLE_CUH__)
#define __TRIANGLE_CUH__

#include "core/ray.cuh"
#include "core/shape.cuh"
#include <math.h>

class TriangleMesh
{
public:
	//int nVertices;
	int nTriangles;
	int* vertexIndices;   // 3 triangles * 3   1 triangle * 3
	int* normalIndices;
	point3* p;
	normal3* n;

	TriangleMesh() = default;
	__device__
	TriangleMesh(const int& nTriangles, const int* vIndices, const int* nIndices, point3* P, normal3* N);
	__device__
	~TriangleMesh();
};


class Triangle : public Shape
{
public:
	TriangleMesh* mesh;
	int v1, v2, v3;   // vertices
	int n1, n2, n3;   // normals

	Triangle() = default;
	__device__
	Triangle(TriangleMesh* mesh, const int& triNumber);
	//__device__
	//Triangle(point3 v1, point3 v2, point3 v3);
	__device__
	~Triangle();
	__device__
	virtual bool hitted_by(const Ray& ray, float& t, float& u, float& v) const override;
	__device__
	virtual normal3 compute_normal_at(const point3& p, const float& u = 0.0f, const float& v = 0.0f) const override;
	__device__
	virtual Shape* get_shape() override { return this; }
};

__device__
Shape** createTriangleMeshShape(const int& nTriangles, int* d_vIndices, int* d_nIndices, point3* d_P, normal3* d_N);


#endif
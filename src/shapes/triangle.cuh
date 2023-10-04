#if !defined(__TRIANGLE_CUH__)
#define __TRIANGLE_CUH__

#include "core/ray.cuh"
#include "core/shape.cuh"
#include <math.h>

class TriangleMesh
{
public:
	int nVertices;
	int nTriangles;
	int vertexIndices[9];   // 3 triangles * 3   1 triangle * 3
	point3 p[4];

	TriangleMesh() = default;
	__device__
	TriangleMesh(const int& nVertices, const int& nTriangles, const int* vIndices, const point3* P);
};


class Triangle : public Shape
{
public:
	TriangleMesh* mesh;
	int v1, v2, v3;

	Triangle() = default;
	__device__
	Triangle(TriangleMesh* mesh, const int& triNumber);
	//__device__
	//Triangle(point3 v1, point3 v2, point3 v3);
	__device__
	~Triangle();
	__device__
	virtual bool hitted_by(const Ray& ray, float& t) const override;
	__device__
	virtual normal3 compute_normal_at(const point3& p) const override;
	__device__
	virtual Shape* get_shape() override { return this; }
};

__device__
Shape** createTriangleMeshShape(const int& nVertices, const int& nTriangles, int* d_vIndices, point3* d_P);


#endif
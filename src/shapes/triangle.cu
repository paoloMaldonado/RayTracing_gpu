#include "triangle.cuh"
#include "utility/utils.cuh"
#include <iostream>

//__device__
//TriangleMesh::TriangleMesh(const int& nVertices, const int& nTriangles, const int* vIndices, const point3* P) :
//	nVertices(nVertices), nTriangles(nTriangles)
//{
//	vertexIndices = new int[nTriangles * 3];
//	p = new point3[nVertices];
//
//	for (int i = 0; i < 3 * nTriangles; ++i) 
//		vertexIndices[i] = vIndices[i];
//
//	for (int i = 0; i < nVertices; ++i) 
//		p[i] = P[i];
//}

__device__
TriangleMesh::TriangleMesh(const int& nTriangles, const int* vIndices, const int* nIndices, point3* P, normal3* N) :
	nTriangles(nTriangles)
{
	vertexIndices = new int[nTriangles * 3];
	normalIndices = new int[nTriangles * 3];

	for (int i = 0; i < 3 * nTriangles; ++i)
	{
		vertexIndices[i] = vIndices[i];
		normalIndices[i] = nIndices[i];
	}

	p = P; // changing ownerwhip of the pointer, P and N are already allocated arrays on global memory
	n = N;
}

__device__
TriangleMesh::~TriangleMesh()
{
	delete[] vertexIndices;
	delete[] normalIndices;
	//delete[] p;
}

__device__
Triangle::Triangle(TriangleMesh* mesh, const int& triNumber) : mesh(mesh)
{
	// these are indices (integers)
	v1 = mesh->vertexIndices[3 * triNumber];
	v2 = mesh->vertexIndices[3 * triNumber + 1];
	v3 = mesh->vertexIndices[3 * triNumber + 2];

	// indices of the vertex normal
	n1 = mesh->normalIndices[3 * triNumber];
	n2 = mesh->normalIndices[3 * triNumber + 1];
	n3 = mesh->normalIndices[3 * triNumber + 2];
}

__device__
Triangle::~Triangle()
{
	// Additionally delete the space reserved for TriangleMesh in Heap just one time 
	// (mesh is always dynamically initialized with new because Triangle only stores a reference to the 
	// TriangleMesh object and not the object itself)
	
	if (mesh != nullptr)
		delete mesh;
}

__device__
bool Triangle::hitted_by(const Ray& ray, float& t, float& u, float& v) const
{
	// access to the elements v1, v2, v3 in the array of points p
	point3 vert0 = mesh->p[v1];
	point3 vert1 = mesh->p[v2];
	point3 vert2 = mesh->p[v3];

	vec3 edge1, edge2, tvec, pvec, qvec;
	float det, inv_det;

	// find vectors for two edges sharing vert
	edge1 = vert1 - vert0;
	edge2 = vert2 - vert0;

	// calculating determinant
	pvec = cross(ray.direction, edge2);

	// if determinant is near zero, ray lies in plane of triangle
	det = dot(edge1, pvec);

	if (fabs(det) < epsilon())
		return false;
	inv_det = 1.0f / det;

	// calculate distance from vert0 to ray origin
	tvec = ray.origin - vert0;

	// calculate u parameter and test bounds
	u = dot(tvec, pvec) * inv_det;
	if (u < 0.0f || u > 1.0f)
		return false;  // out of bounds

	// calculate v parameter and test bounds
	qvec = cross(tvec, edge1);
	v = dot(ray.direction, qvec) * inv_det;
	if (v < 0.0f || u + v > 1.0f)
		return false;

	// calculate t, ray intersects triangle
	t = dot(edge2, qvec) * inv_det;

	return (t > 0.0f) ? true : false;
}

__device__
normal3 Triangle::compute_normal_at(const point3& p, const float& u, const float& v) const
{
	// access to the elements v1, v2, v3 in the array of points p
	point3 vert0 = mesh->p[v1];
	point3 vert1 = mesh->p[v2];
	point3 vert2 = mesh->p[v3];

	if (mesh->n == nullptr)
	{
		vec3 edge1, edge2;
		edge1 = vert1 - vert0;
		edge2 = vert2 - vert0;

		return normal3(normalize(cross(edge2, edge1)));
	}

	// access to the normal vertices of the triangle
	normal3 norm0 = mesh->n[n1];
	normal3 norm1 = mesh->n[n2];
	normal3 norm2 = mesh->n[n3];

	// normal interpolation according to Phong shading -> interpolate the normals with the u v coordinates
	return norm1 * u +
		   norm2 * v +
		   norm0 * (1 - u - v);
}

__device__
Shape** createTriangleMeshShape(const int& nTriangles, int* d_vIndices, int* d_nIndices, point3* d_P, normal3* d_N)
{
	Shape** d_triangles = new Shape*[nTriangles];

	TriangleMesh* mesh = new TriangleMesh(nTriangles, d_vIndices, d_nIndices, d_P, d_N);
	for (int i = 0; i < nTriangles; ++i)
	{
		d_triangles[i] = new Triangle(mesh, i);
	}
	return d_triangles;
}
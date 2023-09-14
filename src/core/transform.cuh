#if !defined(__TRANSFORM_CUH__)
#define __TRANSFORM_CUH__

#include <cuda_runtime.h>
#include "geometry.cuh"
#include "ray.cuh"

class Matrix4x4
{
public:
	float m[4][4];

	__device__
	Matrix4x4()
	{
		m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
		m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] =
			m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
	}
	__device__
	Matrix4x4(float t00, float t01, float t02, float t03, float t10, float t11,
			  float t12, float t13, float t20, float t21, float t22, float t23,
			  float t30, float t31, float t32, float t33);

	__device__
	friend Matrix4x4 transpose(const Matrix4x4& m);

	__device__
	static Matrix4x4 matMul(const Matrix4x4& m1, const Matrix4x4& m2)
	{
		Matrix4x4 r;
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
							m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
			}

		}
		return r;
	}
};

class Transform
{
public:
	Matrix4x4 m;

	Transform() = default;
	__device__
	Transform(const float mat[4][4]) 
	{
		m = Matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
					  mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
					  mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
					  mat[3][3]);
	}
	__device__
	Transform(const Matrix4x4& m) : m(m)
	{}
	__device__
	friend Transform transpose(const Transform& t)
	{
		return Transform(transpose(t.m));
	}
	__device__
	point3 operator()(const point3& p) const;
	__device__
	vec3 operator()(const vec3& v) const;
	__device__
	normal3 operator()(const normal3& n) const;
	__device__
	Ray operator()(const Ray& r) const;
	__device__
	Transform operator*(const Transform& t2) const;
};

__device__
Transform translate(const vec3& delta);
__device__
Transform scaling(const float& x, const float& y, const float& z);
__device__
Transform rotateX(const float& theta);
__device__
Transform rotateY(const float& theta);
__device__
Transform rotateZ(const float& theta);

#endif
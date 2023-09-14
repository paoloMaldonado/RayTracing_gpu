#include "transform.cuh"
#include "math.h"

__device__
Matrix4x4::Matrix4x4(float t00, float t01, float t02, float t03, float t10, float t11,
					 float t12, float t13, float t20, float t21, float t22, float t23,
					 float t30, float t31, float t32, float t33)
{
    m[0][0] = t00;
    m[0][1] = t01;
    m[0][2] = t02;
    m[0][3] = t03;
    m[1][0] = t10;
    m[1][1] = t11;
    m[1][2] = t12;
    m[1][3] = t13;
    m[2][0] = t20;
    m[2][1] = t21;
    m[2][2] = t22;
    m[2][3] = t23;
    m[3][0] = t30;
    m[3][1] = t31;
    m[3][2] = t32;
    m[3][3] = t33;
}

__device__
Matrix4x4 transpose(const Matrix4x4& m)
{
    return Matrix4x4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1],
                     m.m[1][1], m.m[2][1], m.m[3][1], m.m[0][2], m.m[1][2],
                     m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3],
                     m.m[3][3]);
}

__device__
point3 Transform::operator()(const point3& p) const
{
    float x = p.x, y = p.y, z = p.z;
    float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
    float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
    float zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
    return point3(xp, yp, zp);
}

__device__
vec3 Transform::operator()(const vec3& v) const
{
    float x = v.x, y = v.y, z = v.z;
    float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z;
    float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z;
    float zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z;
    return vec3(xp, yp, zp);
}

__device__
normal3 Transform::operator()(const normal3& n) const
{
    float x = n.x, y = n.y, z = n.z;
    float nx = m.m[0][0] * x + m.m[1][0] * y + m.m[2][0] * z;
    float ny = m.m[0][1] * x + m.m[1][1] * y + m.m[2][1] * z;
    float nz = m.m[0][2] * x + m.m[1][2] * y + m.m[2][2] * z;
    return normal3(nx, ny, nz);
}

__device__
Ray Transform::operator()(const Ray& r) const
{
    point3 o = (*this)(r.origin);
    vec3 d   = (*this)(r.direction);
    return Ray(o, d);
}

__device__
Transform Transform::operator*(const Transform& t2) const
{
    return Transform(Matrix4x4::matMul(m, t2.m));
}

__device__
Transform translate(const vec3& delta)
{
    // Inverse translation
    Matrix4x4 m(1, 0, 0, -delta.x,
                0, 1, 0, -delta.y,
                0, 0, 1, -delta.z,
                0, 0, 0, 1);
    return Transform(m);
}

__device__
Transform scaling(const float& x, const float& y, const float& z)
{
    // Inverse scaling
    Matrix4x4 m(1/x, 0, 0, 0,
                0, 1/y, 0, 0,
                0, 0, 1/z, 0,
                0, 0, 0, 1);
    return Transform(m);
}

__device__
Transform rotateX(const float& theta)
{
    // Inverse rotation around the X-axis
    Matrix4x4 m(1, 0, 0, 0,
                0, cosf(theta), sinf(theta), 0,
                0, -sinf(theta), cosf(theta), 0,
                0, 0, 0, 1);
    return Transform(m);
}

__device__
Transform rotateY(const float& theta)
{
    // Inverse rotation around the Y-axis
    Matrix4x4 m(cosf(theta), 0, -sinf(theta), 0,
                0, 1, 0, 0,
                sinf(theta), 0, cosf(theta), 0,
                0, 0, 0, 1);
    return Transform(m);
}

__device__
Transform rotateZ(const float& theta)
{
    // Inverse rotation around the Z-axis
    Matrix4x4 m(cosf(theta), sinf(theta), 0, 0,
                -sinf(theta), cosf(theta), 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
    return Transform(m);
}

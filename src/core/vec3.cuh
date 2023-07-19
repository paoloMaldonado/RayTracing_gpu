#if !defined(__VEC3_CUH__)
#define __VEC3_CUH__

#include <cuda_runtime.h>
#include <math.h>

class vec3
{
public:
	float x;
	float y;
	float z;

	vec3() = default;
	__host__ __device__ 
	vec3(const float& x, const float& y, const float& z) : x(x), y(y), z(z)
	{}
	__host__ __device__
	vec3(const float& v) : x(v), y(v), z(v)
	{}
	__host__ __device__ 
	inline vec3 operator+(const vec3& b) const { return vec3(x + b.x, y + b.y, z + b.z); }
	__host__ __device__
	inline vec3 operator/(const float& d) const { return vec3(x / d, y / d, z / d); }
	__host__ __device__
	inline vec3 operator-(const vec3& b) const { return vec3(x - b.x, y - b.y, z - b.z); }
	__host__ __device__
	inline vec3 operator-() const { return vec3(-x, -y, -z); }
	__host__ __device__
	inline vec3 operator*(const vec3& b) const { return vec3(x*b.x, y*b.y, z*b.z); }
	__host__ __device__
	inline vec3& operator+=(const vec3& b);
	__host__ __device__
	inline vec3& operator-=(const vec3& b);
	__host__ __device__
	inline vec3& operator*=(const vec3& b);
	__host__ __device__
	inline float lenght() const { return sqrtf(x*x + y*y + z*z); }
	__host__ __device__
	inline float* value_ptr() { return &(x); }
};

__host__ __device__
inline vec3& vec3::operator+=(const vec3& b)
{
	x += b.x;
	y += b.y;
	z += b.z;
	return *this;
}

__host__ __device__
inline vec3& vec3::operator-=(const vec3& b)
{
	x -= b.x;
	y -= b.y;
	z -= b.z;
	return *this;
}

__host__ __device__
inline vec3& vec3::operator*=(const vec3& b)
{
	x *= b.x;
	y *= b.y;
	z *= b.z;
	return *this;
}

__host__ __device__
inline vec3 operator*(const vec3& a, const float& scalar) { return vec3(scalar * a.x, scalar * a.y, scalar * a.z); }

__host__ __device__
inline vec3 operator*(const float& scalar, const vec3& a) { return vec3(scalar * a.x, scalar * a.y, scalar * a.z); }

__host__ __device__
inline vec3 normalize(const vec3& vec) 
{
	float l = vec.lenght();
	return vec / l;
}

__host__ __device__
inline float dot(const vec3& a, const vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__host__ __device__
inline vec3 cross(const vec3& a, const vec3& b) {
	return vec3((a.y * b.z - a.z * b.y),
		(-(a.x * b.z - a.z * b.x)),
		(a.x * b.y - a.y * b.x));
}

#endif
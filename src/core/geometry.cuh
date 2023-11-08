#if !defined(__GEOMETRY_CUH__)
#define __GEOMETRY_CUH__

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
	inline vec3 operator-(const vec3& b) const { return vec3(x - b.x, y - b.y, z - b.z); }
	__host__ __device__
	inline vec3 operator*(const vec3& b) const { return vec3(x * b.x, y * b.y, z * b.z); }
	__host__ __device__
	inline vec3 operator/(const float& d) const { return vec3(x / d, y / d, z / d); }
	__host__ __device__
	inline vec3 operator-() const { return vec3(-x, -y, -z); }
	__host__ __device__
	inline vec3& operator+=(const vec3& b)
	{
		x += b.x;
		y += b.y;
		z += b.z;
		return *this;
	}
	__host__ __device__
	inline vec3& operator-=(const vec3& b)
	{
		x -= b.x;
		y -= b.y;
		z -= b.z;
		return *this;
	}
	__host__ __device__
	inline vec3& operator*=(const vec3& b)
	{
		x *= b.x;
		y *= b.y;
		z *= b.z;
		return *this;
	}
	__host__ __device__
	inline vec3& operator/=(const vec3& b)
	{
		x /= b.x;
		y /= b.y;
		z /= b.z;
		return *this;
	}
	__host__ __device__
	inline float lenght() const { return sqrtf(x*x + y*y + z*z); }
	__host__ __device__
	inline float* value_ptr() { return &(x); }
};

class normal3
{
public:
	float x;
	float y;
	float z;

	normal3() = default;
	__host__ __device__
	normal3(const float& x, const float& y, const float& z) : x(x), y(y), z(z)
	{}
	__host__ __device__
	normal3(const float& v) : x(v), y(v), z(v)
	{}
	__host__ __device__
	normal3(const vec3& v) : x(v.x), y(v.y), z(v.z)
	{}
	__host__ __device__
	inline normal3 operator+(const normal3& b) const { return normal3(x + b.x, y + b.y, z + b.z); }
	__host__ __device__
	inline normal3 operator-(const normal3& b) const { return normal3(x - b.x, y - b.y, z - b.z); }
	__host__ __device__
	inline normal3 operator*(const normal3& b) const { return normal3(x * b.x, y * b.y, z * b.z); }
	__host__ __device__
	inline normal3 operator/(const float& d) const { return normal3(x / d, y / d, z / d); }
	__host__ __device__
	inline normal3 operator-() const { return normal3(-x, -y, -z); }
	__host__ __device__
	inline normal3& operator+=(const normal3& b)
	{
		x += b.x;
		y += b.y;
		z += b.z;
		return *this;
	}
	__host__ __device__
	inline normal3& operator-=(const normal3& b)
	{
		x -= b.x;
		y -= b.y;
		z -= b.z;
		return *this;
	}
	__host__ __device__
	inline normal3& operator*=(const normal3& b)
	{
		x *= b.x;
		y *= b.y;
		z *= b.z;
		return *this;
	}
	__host__ __device__
	inline normal3& operator/=(const normal3& b)
	{
		x /= b.x;
		y /= b.y;
		z /= b.z;
		return *this;
	}
	__host__ __device__
	inline float lenght() const { return sqrtf(x * x + y * y + z * z); }
	__host__ __device__
	inline float* value_ptr() { return &(x); }
};

class point3
{
public:
	float x;
	float y;
	float z;

	point3() = default;
	__host__ __device__
	point3(const float& x, const float& y, const float& z) : x(x), y(y), z(z)
	{}
	__host__ __device__
	point3(const float& v) : x(v), y(v), z(v)
	{}
	__host__ __device__
	inline point3 operator+(const vec3& b) const { return point3(x + b.x, y + b.y, z + b.z); }
	__host__ __device__
	inline vec3 operator-(const point3& b) const { return vec3(x - b.x, y - b.y, z - b.z); }
	__host__ __device__
	inline point3 operator-(const vec3& b) const { return point3(x - b.x, y - b.y, z - b.z); }
	__host__ __device__
	inline point3 operator*(const point3& b) const { return point3(x * b.x, y * b.y, z * b.z); }
	__host__ __device__
	inline point3 operator/(const float& d) const { return point3(x / d, y / d, z / d); }
	__host__ __device__
	inline point3 operator-() const { return point3(-x, -y, -z); }
	__host__ __device__
	inline point3& operator+=(const vec3& b)
	{
		x += b.x;
		y += b.y;
		z += b.z;
		return *this;
	}
	__host__ __device__
	inline point3& operator-=(const vec3& b)
	{
		x -= b.x;
		y -= b.y;
		z -= b.z;
		return *this;
	}
	__host__ __device__
	inline point3& operator*=(const point3& b)
	{
		x *= b.x;
		y *= b.y;
		z *= b.z;
		return *this;
	}
	__host__ __device__
	inline point3& operator/=(const point3& b)
	{
		x /= b.x;
		y /= b.y;
		z /= b.z;
		return *this;
	}
	__host__ __device__
	inline float* value_ptr() { return &(x); }
};

class Spectrum
{
public:
	float x;
	float y;
	float z;

	Spectrum() = default;
	__host__ __device__
	Spectrum(const float& x, const float& y, const float& z) : x(x), y(y), z(z)
	{}
	__host__ __device__
	Spectrum(const float& v) : x(v), y(v), z(v)
	{}
	__host__ __device__
	Spectrum(const float v[3]) : x(v[0]), y(v[1]), z(v[2])
	{}
	__host__ __device__
	inline Spectrum operator+(const Spectrum& b) const { return Spectrum(x + b.x, y + b.y, z + b.z); }
	__host__ __device__
	inline Spectrum operator-(const Spectrum& b) const { return Spectrum(x - b.x, y - b.y, z - b.z); }
	__host__ __device__
	inline Spectrum operator*(const Spectrum& b) const { return Spectrum(x * b.x, y * b.y, z * b.z); }
	__host__ __device__
	inline Spectrum operator/(const float& d) const { return Spectrum(x / d, y / d, z / d); }
	__host__ __device__
	inline Spectrum operator-() const { return Spectrum(-x, -y, -z); }
	__host__ __device__
	inline Spectrum& operator+=(const Spectrum& b)
	{
		x += b.x;
		y += b.y;
		z += b.z;
		return *this;
	}
	__host__ __device__
	inline Spectrum& operator-=(const Spectrum& b)
	{
		x -= b.x;
		y -= b.y;
		z -= b.z;
		return *this;
	}
	__host__ __device__
	inline Spectrum& operator*=(const Spectrum& b)
	{
		x *= b.x;
		y *= b.y;
		z *= b.z;
		return *this;
	}
	__host__ __device__
	inline Spectrum& operator/=(const Spectrum& b)
	{
		x /= b.x;
		y /= b.y;
		z /= b.z;
		return *this;
	}
	__host__ __device__
	inline float* value_ptr() { return &(x); }
	__host__ __device__
	inline bool isBlack() { return (x == 0.0f && y == 0.0f && z == 0.0f) ? true : false; }
};


__host__ __device__
inline vec3 operator-(const normal3& n, const vec3& v) { return vec3(n.x - v.x, n.y - v.y, n.z - v.z); }
__host__ __device__
inline vec3 operator-(const vec3& v, const normal3& n) { return vec3(v.x - n.x, v.y - n.y, v.z - n.z); }

__host__ __device__
inline vec3 operator+(const normal3& n, const vec3& v) { return vec3(n.x + v.x, n.y + v.y, n.z + v.z); }
__host__ __device__
inline vec3 operator+(const vec3& v, const normal3& n) { return vec3(v.x + n.x, v.y + n.y, v.z + n.z); }

__host__ __device__
inline vec3 operator*(const vec3& a, const float& scalar) { return vec3(scalar * a.x, scalar * a.y, scalar * a.z); }
__host__ __device__
inline vec3 operator*(const float& scalar, const vec3& a) { return vec3(scalar * a.x, scalar * a.y, scalar * a.z); }

__host__ __device__
inline point3 operator*(const point3& a, const float& scalar) { return point3(scalar * a.x, scalar * a.y, scalar * a.z); }
__host__ __device__
inline point3 operator*(const float& scalar, const point3& a) { return point3(scalar * a.x, scalar * a.y, scalar * a.z); }

__host__ __device__
inline normal3 operator*(const normal3& a, const float& scalar) { return normal3(scalar * a.x, scalar * a.y, scalar * a.z); }
__host__ __device__
inline normal3 operator*(const float& scalar, const normal3& a) { return normal3(scalar * a.x, scalar * a.y, scalar * a.z); }

__host__ __device__
inline Spectrum operator*(const Spectrum& a, const float& scalar) { return Spectrum(scalar * a.x, scalar * a.y, scalar * a.z); }
__host__ __device__
inline Spectrum operator*(const float& scalar, const Spectrum& a) { return Spectrum(scalar * a.x, scalar * a.y, scalar * a.z); }

__host__ __device__
inline vec3 normalize(const vec3& vec) 
{
	float l = vec.lenght();
	return vec / l;
}

__host__ __device__
inline normal3 normalize(const normal3& vec)
{
	float l = vec.lenght();
	return vec / l;
}

// Dot product
__host__ __device__
inline float dot(const vec3& a, const vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__host__ __device__
inline float dot(const normal3& a, const normal3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__host__ __device__
inline float dot(const normal3& a, const vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__host__ __device__
inline float dot(const vec3& a, const normal3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Cross product
__host__ __device__
inline vec3 cross(const vec3& a, const vec3& b) {
	return vec3((a.y * b.z - a.z * b.y),
		(-(a.x * b.z - a.z * b.x)),
		(a.x * b.y - a.y * b.x));
}

__host__ __device__
inline vec3 cross(const normal3& a, const normal3& b) {
	return vec3((a.y * b.z - a.z * b.y),
		(-(a.x * b.z - a.z * b.x)),
		(a.x * b.y - a.y * b.x));
}

__host__ __device__
inline vec3 cross(const normal3& a, const vec3& b) {
	return vec3((a.y * b.z - a.z * b.y),
		(-(a.x * b.z - a.z * b.x)),
		(a.x * b.y - a.y * b.x));
}

__host__ __device__
inline vec3 cross(const vec3& a, const normal3& b) {
	return vec3((a.y * b.z - a.z * b.y),
		(-(a.x * b.z - a.z * b.x)),
		(a.x * b.y - a.y * b.x));
}

#endif
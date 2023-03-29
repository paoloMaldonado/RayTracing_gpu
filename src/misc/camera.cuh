#if !defined(__CAMERA_CUH__)
#define __CAMERA_CUH__

#include "vec3.cuh"
#include <math_constants.h>

class Camera
{
public:
	// right handed camera (parameters expressed in world space)
	vec3 e;			// camera origin / center of projection
	vec3 front;		// camera looking direction
	vec3 up;		// up vector (0, 1, 0) by deafult
	vec3 u, v, w;	// view space orthonormal basis 
	float d;		// focal length / distance to view plane

	float yaw;
	float pitch;

	Camera() = default;
	__host__ 
	Camera(const vec3& eye, const float& d, const vec3& front = vec3(0.0f, 0.0f, -1.0f), const float& yaw = 90.0f, const float& pitch = -20.0f) : e(eye), d(d), front(front), yaw(yaw), pitch(pitch)
	{
		up = vec3(0.0f, 1.0f, 0.0f);
		u = vec3(1.0f, 0.0f, 0.0f);
		v = vec3(0.0f, 1.0f, 0.0f);
		w = vec3(0.0f, 0.0f, 1.0f);

		set_pitch_angle(pitch);
		set_yaw_angle(yaw);
	}

	__host__ 
	inline void set_pitch_angle(const float& pitch_angle) 
	{ 
		pitch = to_radians(pitch_angle);
	}

	__host__
	inline void set_yaw_angle(const float& yaw_angle)
	{
		yaw = to_radians(yaw_angle);
	}

	__device__
	inline void compute_view_basis()
	{
		vec3 look_at = e + front;
		w = e - look_at;
		w.x = cosf(pitch) * cosf(yaw);
		w.y = sinf(pitch);
		w.z = cosf(pitch) * sinf(yaw);

		w = normalize(w);
		u = normalize(cross(up, w));
		v = cross(w, u);
	}

	__host__ __device__
	inline void translate(vec3 translation) { e += translation; }

private:
	__host__
	inline float to_radians(const float& angle) const
	{
		return angle * (CUDART_PI_F / 180.0f);
	}
};

#endif
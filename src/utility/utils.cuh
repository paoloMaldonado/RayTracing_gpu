#if !defined(__UTILS_GPU_CUH__)
#define __UTILS_GPU_CUH__

#include "core/geometry.cuh"

__device__
inline float pi() { return 3.14159265358979323846f; }
__device__
inline float twopi() { return 6.28318530717958647692f; }
__device__
inline float invPi() { return 0.31830988618379067154f; }
__device__
inline float epsilon() { return 0.0001f; }

template <class T>
__device__ void swap(T& a, T& b)
{
	T c(a); a = b; b = c;
}

__device__
inline bool areEqualf(const float& a, const float& b, const float& epsilon = 0.0001f)
{
	return (fabsf(a - b) < epsilon) ? true : false;
}

__device__ 
inline float clamp(float f, float a, float b)
{
	return fmaxf(a, fminf(f, b));
}

// from: https://stackoverflow.com/a/65419193
__device__
inline int cuda_strcmp(const char* s1, const char* s2) 
{
    for (; *s1 == *s2; ++s1, ++s2)
        if (*s1 == 0)
            return 0;
    return *(unsigned char*)s1 < *(unsigned char*)s2 ? -1 : 1;
}

#endif
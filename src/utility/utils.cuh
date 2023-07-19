#if !defined(__UTILS_GPU_CUH__)
#define __UTILS_GPU_CUH__

#include "core/vec3.cuh"

__device__
inline float pi() { return 3.14159265358979323846f; }
__device__
inline float twopi() { return 6.28318530717958647692f; }
__device__
inline float invPi() { return 0.31830988618379067154f; }

#endif
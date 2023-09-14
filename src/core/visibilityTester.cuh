#if !defined(__VISIBILITY_TESTER_CUH__)
#define __VISIBILITY_TESTER_CUH__

#include "kernels/intersection_kernel.cuh"

class VisibilityTester
{
public:
	Instance** escene;
	int N;

	VisibilityTester() = default;
	__device__
	VisibilityTester(Instance** escene, const int& N) : 
		escene(escene), N(N)
	{}
	__device__
	bool test_shadow(const Ray& shadow_ray) const
	{
		// create shadow ray
		return intersectionShadow(shadow_ray, escene, N);
	}
};

#endif
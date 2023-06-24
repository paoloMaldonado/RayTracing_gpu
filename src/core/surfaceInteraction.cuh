#if !defined(__SURFACEINTERACTION_CUH__)
#define __SURFACEINTERACTION_CUH__

#include "shapes/sphere.cuh"

class SurfaceInteraction
{
public:
	float t;
	Sphere hitobject;

	SurfaceInteraction() = default;
};

#endif
#pragma once

#include <cmath>

inline bool areEqual(const float& a, const float& b, const float& epsilon = 0.0001f)
{
	return (fabs(a - b) < epsilon) ? true : false;
}


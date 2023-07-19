#if !defined(__MEMORY_CUH__)
#define __MEMORY_CUH__

#include <device_launch_parameters.h>

#define ALLOC(mem, Type) new ((mem).alloc(sizeof(Type))) Type

class MemoryManager
{
public:
	unsigned char __align__(8) buffer[1024];
	size_t memory_block = 0;

	MemoryManager() = default;
	__device__
	unsigned char* alloc(const size_t& byte_size)
	{
		unsigned char* b_temp = buffer + memory_block;
		memory_block += byte_size;
		return b_temp;
	}
};

#endif
#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class GraphicsResourceFromGL
{
public:
	GraphicsResourceFromGL() = default;
	GraphicsResourceFromGL(unsigned int opengl_object_id) : mappedPointer(nullptr), numBytes(0)
	{
		cudaGraphicsGLRegisterBuffer(&graphicsResource, opengl_object_id, cudaGraphicsMapFlagsWriteDiscard);
	}
	inline float4* mapAndReturnDevicePointer() 
	{
		cudaGraphicsMapResources(1, &graphicsResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&mappedPointer, &numBytes, graphicsResource);
		return mappedPointer;
	}
	inline void unmap()
	{
		cudaGraphicsUnmapResources(1, &graphicsResource, 0);
	}

private:
	cudaGraphicsResource* graphicsResource;
	float4* mappedPointer;
	size_t numBytes;
};


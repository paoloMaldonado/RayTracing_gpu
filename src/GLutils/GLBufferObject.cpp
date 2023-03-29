#include "GLBufferObject.h"

GLBufferObject::GLBufferObject(const GLenum& target, const void* data, const GLsizeiptr& size, const GLenum& usage) : 
	ID(0), target(target)
{
	if (ID == 0)
		glGenBuffers(1, &ID);

	glBindBuffer(target, ID);
	glBufferData(target, size, data, usage);
	glBindBuffer(target, 0);
}

GLBufferObject::~GLBufferObject()
{
	glBindBuffer(target, 0);
}

void GLBufferObject::bind()
{
	glBindBuffer(target, ID);
}

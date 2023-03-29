#pragma once
#include <glad/glad.h>

class GLBufferObject
{
public:
	GLuint ID;
	GLenum target;

	GLBufferObject() = default;
	GLBufferObject(const GLenum& target, const void* data, const GLsizeiptr& size, const GLenum& usage = GL_STATIC_DRAW);
	~GLBufferObject();
	void bind();
};


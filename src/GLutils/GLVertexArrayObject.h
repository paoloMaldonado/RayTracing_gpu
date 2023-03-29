#pragma once
#include <glad/glad.h>
#include "GLBufferObject.h"
#include <vector>

struct GLVertexAttribute
{
	GLuint index;
	GLint size;
	GLenum type;
	GLsizei stride;
	const void* pointer;
};

class GLVertexArrayObject
{
public:
	GLuint ID;

	GLVertexArrayObject() = default;
	GLVertexArrayObject(GLBufferObject& bufferToBind, const std::vector<GLVertexAttribute>& vAttribs);
	~GLVertexArrayObject();
	void bind();
};


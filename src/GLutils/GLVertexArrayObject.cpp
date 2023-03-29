#include "GLVertexArrayObject.h"

GLVertexArrayObject::GLVertexArrayObject(GLBufferObject& bufferToBind, const std::vector<GLVertexAttribute>& vAttribs) : ID(0)
{
	if (ID == 0)
		glGenVertexArrays(1, &ID);
	glBindVertexArray(ID);

	bufferToBind.bind();

	for (auto& attrib : vAttribs)
	{
		glVertexAttribPointer(attrib.index, attrib.size, attrib.type, GL_FALSE, attrib.stride, attrib.pointer);
		glEnableVertexAttribArray(attrib.index);
	}

	glBindVertexArray(0);
}

GLVertexArrayObject::~GLVertexArrayObject()
{
	glBindVertexArray(0);
}

void GLVertexArrayObject::bind()
{
	glBindVertexArray(ID);
}

#pragma once
#include <glad/glad.h>

class GLTexture
{
public:
	GLuint ID;
	GLenum target;
	GLenum internalFormat;
	GLenum format;
	GLenum type;
	GLsizei width;
	GLsizei height;

	GLTexture() = default;
	GLTexture(const GLenum& target, const GLenum& internalFormat, const GLenum& format, 
		const GLenum& type, const GLsizei& width, const GLsizei& height);
	~GLTexture();
	void loadTexture(const void* data);
};


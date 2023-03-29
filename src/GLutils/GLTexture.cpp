#include "GLTexture.h"

GLTexture::GLTexture(const GLenum& target, const GLenum& internalFormat, const GLenum& format,
					 const GLenum& type, const GLsizei& width, const GLsizei& height) :
	ID(0), target(target), internalFormat(internalFormat), format(format), type(type), width(width), height(height)
{
	if (ID == 0)
		glGenTextures(1, &ID);

	glBindTexture(target, ID);

	switch (target)
	{
	case GL_TEXTURE_2D:
		glTexStorage2D(target, 
					   1, 
					   internalFormat, 
			           width, 
			           height);
	default:
		break;
	}

	glBindTexture(target, 0);
}

GLTexture::~GLTexture()
{
	glBindTexture(target, 0);
}

void GLTexture::loadTexture(const void* data)
{
	glBindTexture(target, ID);
	glTexSubImage2D(target,			           // target
					0,						   // First mipmap level                           
					0, 0,					   // x and y offset
					width, height,	           // width and height
					format, type,		       // external format and type    
					data);					   // data
}

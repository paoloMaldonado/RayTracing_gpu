#pragma once
#include <glad/glad.h>
#include <shader.h>
#include "GLutils/GLTexture.h"
#include "GLutils/GLVertexArrayObject.h"

class Canvas
{
public:
	Canvas() = default;
	Canvas(const unsigned int& WIDTH, const unsigned int& HEIGHT);
	void renderTexture();
	inline GLuint getPixelDataID() { return pixelData.ID; };
	inline GLuint getTextureWallID() { return canvasWall.ID; };

private:
	static const GLfloat quad_data[];
	Shader shader;
	GLBufferObject pixelData;
	GLTexture canvasWall;
	GLVertexArrayObject vaoForDrawing;
};


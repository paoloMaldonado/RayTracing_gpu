#include "Canvas.h"

const GLfloat Canvas::quad_data[] = 
{
    // Vertex positions 
    -1.0f, -1.0f, 0.0f, 1.0f,
     1.0f, -1.0f, 0.0f, 1.0f,
     1.0f,  1.0f, 0.0f, 1.0f,
    -1.0f,  1.0f, 0.0f, 1.0f,
    // Texture coordinates
    0.0f, 0.0f,
    1.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 1.0f
};

// Compile the shader, this shader makes possible the texture sampling into the quad  
//Shader Canvas::shader("shaders/vertex_shader.vert", "shaders/fragment_shader.frag");

Canvas::Canvas(const unsigned int& WIDTH, const unsigned int& HEIGHT)
{
    this->shader = Shader("shaders/vertex_shader.vert", "shaders/fragment_shader.frag");

    // 1. Upload quad_data to a vertex buffer object (VBO) in GPU
    GLBufferObject quad(GL_ARRAY_BUFFER, quad_data, sizeof(quad_data));

    // 2. Set a VAO, then config the vertex layout (the order in which the gpu will make use of its data)
    GLVertexAttribute attrPosition{ 0, 4, GL_FLOAT, 0, 0 };
    GLVertexAttribute attrTexCoords{ 1, 2, GL_FLOAT, 0, (void*)(16 * sizeof(float))};
    std::vector<GLVertexAttribute> attribs{ attrPosition, attrTexCoords };
    GLVertexArrayObject vao(quad, attribs);
    this->vaoForDrawing = vao;

    // 3. Set a pixel unpack buffer which will hold the pixel data to feed the texture memory later
    // notice how we allocate a pixel buffer with the same size and dimensions as the texture 
    unsigned int size = HEIGHT * WIDTH * 4 * sizeof(float);
    GLBufferObject pixelData(GL_PIXEL_UNPACK_BUFFER, 0, size);
    this->pixelData = pixelData;

    // 4. Allocate a texture in GPU memory with an offset to the pixel buffer previously allocated, here
    // we are only allocating a "wall" of size width x heigth with each pixel being a RGBA vector(float4) type
    this->pixelData.bind();
    GLTexture wall(GL_TEXTURE_2D, GL_RGBA32F, GL_RGBA, GL_FLOAT, WIDTH, HEIGHT);
    this->canvasWall = wall;
}

void Canvas::renderTexture()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    shader.use();

    pixelData.bind();
    canvasWall.loadTexture(0);

    vaoForDrawing.bind();
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
}

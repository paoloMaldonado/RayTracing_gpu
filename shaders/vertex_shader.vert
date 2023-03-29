#version 430 core
layout (location = 0) in vec4 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 tex_coord;

void main()
{
    gl_Position = aPos;
    tex_coord = aTexCoords;
}
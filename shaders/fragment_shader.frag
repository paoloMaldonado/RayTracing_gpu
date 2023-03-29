#version 430 core
uniform sampler2D tex;

in vec2 tex_coord;
out vec4 color;

void main()
{
    color = texture(tex, tex_coord);
}
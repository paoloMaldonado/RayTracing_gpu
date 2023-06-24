#version 430 core
uniform sampler2D tex;

in vec2 tex_coord;
out vec4 color;

//vec3 sRGB(vec3 x) 
//{
//    if (lessThan(x, vec3(0.00031308)) == bvec3(true))
//        return 12.92 * x;
//    else
//        return 1.055 * pow(x, vec3(1./2.4)) - 0.055;
//}

void main()
{
    float gamma = 2.2f;
    vec3 tex = pow(texture(tex, tex_coord).rgb, vec3(1.0/gamma));
    color = vec4(tex, 1.0f);

//    color = texture(tex, tex_coord);

//    color = vec4(sRGB(texture(tex, tex_coord).rgb), 1.0f);
}
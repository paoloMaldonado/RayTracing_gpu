# Whitted ray tracing using GPU acceleration

![ray-tracing-gpu](https://i.imgur.com/GBDb0u9.png)

I implemented ray tracing entirely on a GPU based on Whitted’s algorithm, to render fully ray-traced scenes in real-time settings through a user interface. This project served as my undergraduate thesis. The implementation covers ray tracing for perfect light reflection and refraction, as outlined in the original paper. The algorithm is capable of reproducing light effects such as mirror reflection and diffuse reflection. Additionally, I implemented five Physically Based Rendering (PBR) materials using a design pattern based on the PBRT book. The implementation also includes linear transformations, scene loading, and real-time configuration. Users can explore and experiment with scenes, adjusting settings such as the number of bounces, light, and camera position.

## Dependencies
- Install the CUDA Toolkit from the [official Nvidia page](https://developer.nvidia.com/cuda-toolkit)
- Any video card compatible with OpenGL >=4.3

## To run
Open ./RayTracing_gpu.sln for compiling and debugging. 

Tested with Visual Studio 2022 (64-bit) Version 17.1.5 and CUDA 12.0

Compile in Release mode to use the full potential in performance

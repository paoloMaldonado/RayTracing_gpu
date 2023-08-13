#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif

#include "imguiWidgets/ImguiCustomWidgets.h"

#include <iostream>
#include <map>
#include <vector>

#include <glad/glad.h>
#include "render/WindowHandler.h"
#include "render/Canvas.h"
#include "core/scene.cuh"
#include "render/GraphicsResourceFromGL.h"
#include "input/KeyInput.h"
#include "input/MouseInput.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cudaRaytracing.cuh"

void processKeyInputsForCamera(const std::vector<int>& keys, KeyInput& keyInput, Camera& camera);

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;
float speed = 1.0f;

int main()
{
    const float aspect_ratio = 16.0f / 9.0f;
    const int SCR_WIDTH = 1024;
    const int SCR_HEIGHT = static_cast<int>(SCR_WIDTH / aspect_ratio);

    // set 1.0f to be in front of the shapes (smaller projection), values > 1.0f set the view plane behind the shapes (bigger projection)
    // this phenomenom is due to the center of projection being located at the camera origin (and also because perspective projection works like that)
    Camera camera(vec3(0.0f, 0.0f, 1.0f), 1.0f); 

    WindowHandler window_handler(SCR_WIDTH, SCR_HEIGHT, "raytracing");
    window_handler.mark_as_current_context();

    // Set the Keyboard Input, defining keys for moving around the scene as well as "mod" keys for toggle the imgui panel 
    std::vector<int> keys = { GLFW_KEY_RIGHT,
                              GLFW_KEY_LEFT, 
                              GLFW_KEY_UP, 
                              GLFW_KEY_DOWN };

    std::vector<int> mods = { GLFW_MOD_ALT_X };
    KeyInput keyInput(keys, mods);
    KeyInput::setKeyboardInput(window_handler.window);

    MouseInput mouseInput(0.005f, SCR_WIDTH, SCR_HEIGHT, camera.pitch, camera.yaw);
    MouseInput::setMouseInput(window_handler.window);

    /////////////////////////////////////////////////////////////////////////////////////////

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    const char* glsl_version = "#version 430";
    ImGui_ImplGlfw_InitForOpenGL(window_handler.window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    //////////////////////////////////////////////////////////////////////////////////////

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //glEnable(GL_FRAMEBUFFER_SRGB);
    Canvas canvas(SCR_WIDTH, SCR_HEIGHT);

    // interop between buffer unpack and CUDA
    cudaGraphicsResource* tex_data_resource;
    cudaGraphicsGLRegisterBuffer(&tex_data_resource, canvas.getPixelDataID(), cudaGraphicsMapFlagsWriteDiscard);

    // Graphics settings

    vec3 light(8.5f, 5.5f, 10.0f);

    //Colors
    //vec3 RED = vec3(1.0f, 0.0f, 0.0f);
    //vec3 GREEN = vec3(0.0f, 1.0f, 0.0f);
    //vec3 BLUE = vec3(0.0f, 0.0f, 1.0f);
    //vec3 BLACK = vec3(0.0f, 0.0f, 0.0f);

    //Material mat1(RED, vec3(0.5f, 0.5, 0.5f), 200.0f);
    //Material mat2(GREEN, vec3(0.5f, 0.5, 0.5f), 200.0f);
    //Material mat3(BLACK, vec3(0.5f, 0.5, 0.5f), 200.0f);

    //Sphere object_1(vec3(0.0f, 0.0f, -1.0f), 0.5f, mat1);
    //Sphere object_2(vec3(-1.0f, 0.0f, -1.0f), 0.5f, mat2);
    //Sphere object_3(vec3(1.0f, 0.0f, -1.0f), 0.5f, mat3);
    //Sphere object_4(vec3(0.0f, -100.5f, -1.0f), 100.0f, mat2);

    //std::vector<Sphere> spheres = {object_1, object_2, object_3, object_4};

    Shape** spheres = nullptr;
    Material** materials = nullptr;

    Scene scene(spheres, 4, materials, 3); //instantiate 4 spheres on GPU
    scene.build();

    // render loop
    while (!glfwWindowShouldClose(window_handler.window))
    {
        // compute delta time 
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // compute view-space basis on CPU
        camera.compute_view_basis();

        // CUDA
        float4* dptr = nullptr;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &tex_data_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, tex_data_resource);
        
        //float4* dptr = graphicsResource.mapAndReturnDevicePointer();

        // launch kernel
        callRayTracingKernel(dptr, scene.d_objects, scene.primitive_count, camera, light, SCR_WIDTH, SCR_HEIGHT);

        cudaGraphicsUnmapResources(1, &tex_data_resource, 0);
        //graphicsResource.unmap();

        //input
        glfwPollEvents();
        processKeyInputsForCamera(keys, keyInput, camera);
        bool showWindow = keyInput.isModPressed(GLFW_KEY_X, GLFW_MOD_ALT);
        mouseInput.pointerMode(window_handler.window, showWindow);

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::ShowPerfomanceMetrics();

        if(showWindow)
        {
            ImGui::Begin("Camera");
            ImGui::DragFloat3("Front", camera.front.value_ptr(), 0.1f, -10.0f, 10.0f);
            ImGui::DragFloat3("Position", camera.e.value_ptr(), 0.1f, -10.0f, 10.0f);
            ImGui::DragFloat("Yaw angle", &camera.yaw, 0.1f, -90.0f, 90.0f);
            ImGui::DragFloat("Pitch angle", &camera.pitch, 0.1f, -90.0f, 90.0f);
            ImGui::End();

            ImGui::Begin("Light");
            ImGui::DragFloat3("Position", light.value_ptr(), 0.1f, -10.0f, 10.0f);
            ImGui::End();
        }

        ImGui::Render();

        canvas.renderTexture();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_handler.window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    scene.destroy();
}


void processKeyInputsForCamera(const std::vector<int>& keys, KeyInput& keyInput, Camera& camera)
{
    for (const auto& key : keys)
    {
        switch (key)
        {
        case GLFW_KEY_RIGHT:
            if (keyInput.isKeyDown(key)) camera.translate(cross(camera.front, camera.up) * speed * deltaTime);
            break;
        case GLFW_KEY_LEFT:
            if (keyInput.isKeyDown(key)) camera.translate(-cross(camera.front, camera.up) * speed * deltaTime);
            break;
        case GLFW_KEY_UP:
            if (keyInput.isKeyDown(key)) camera.translate(camera.front * speed * deltaTime);
            break;
        case GLFW_KEY_DOWN:
            if (keyInput.isKeyDown(key)) camera.translate(-camera.front * speed * deltaTime);
            break;
        default:
            break;
        }
    }
}


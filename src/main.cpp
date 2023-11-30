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

#include "render/Loader.h"

void processKeyInputsForCamera(const std::vector<int>& keys, KeyInput& keyInput, Camera& camera);

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;
float speed = 1.0f;

int main()
{
    const float aspect_ratio = 16.0f / 9.0f;
    const int SCR_WIDTH = 1600;
    const int SCR_HEIGHT = static_cast<int>(SCR_WIDTH / aspect_ratio);

    std::cout << "Resolution: " << "\n"<<"\t"<<SCR_WIDTH<<" x "<<SCR_HEIGHT<<"\n";

    // set 1.0f to be in front of the shapes (smaller projection), values > 1.0f set the view plane behind the shapes (bigger projection)
    // this phenomenom is due to the center of projection being located at the camera origin (and also because perspective projection works like that)
    Camera camera(point3(0.012f, 0.700f, 3.043f), 3.0f);   // 0.012f, 0.700f, 3.043f

    // For cornellBoxOriginal -> 3.271f, 0.700f, 2.614f
    // For CornellBoxSphere   -> 0.012f, 0.700f, 3.043f
    // For bunny              -> -0.294f, 1.0f, 2.609f
    // For teapot             -> -0.041f, 1.7f, 8.097f

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

    //point3 light1(8.5f, 5.5f, 10.0f);
    //point3 light2(-8.5f, 5.5f, 10.0f);

    point3 light1(-0.3f, 0.8f, 1.8f);
    point3 light2(0.6f, 5.9f, 9.3f);

    //instantiate 4 spheres on GPU
    Scene scene(4,              // number of primitives (Does not have effect if load_to_gpu() is used)
                4);             // number of materials to allocate

    scene.load_obj_to_gpu("objects/CornellBox-Original.obj", "objects");
    scene.build();

    // set number of bounces
    int depth = 1;

    ////////////////////////////////////////////////////////
    // for calculating number of rays
    //int* primary_rays;
    //int* secondary_rays;

    //int N = SCR_WIDTH * SCR_HEIGHT;
    //cudaMallocManaged(&primary_rays, N * sizeof(int));
    //cudaMallocManaged(&secondary_rays, N * 7 * sizeof(int));

    //for (int i = 0; i < N; i++) { primary_rays[i] = 0; }
    //for (int i = 0; i < N*7; i++) { secondary_rays[i] = 0; }
    //
    //int primary_total = 0;
    //int secondary_total = 0;
    ////////////////////////////////////////////////////////
     
   
    // render loop
    while (!glfwWindowShouldClose(window_handler.window))
    {
        //for (int i = 0; i < N; i++) { primary_rays[i] = 0; }
        //for (int i = 0; i < N*7; i++) { secondary_rays[i] = 0; }

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
        callRayTracingKernel(dptr, scene.get_pointer_to_instances(), scene.number_of_primitives(), camera, light1, light2, depth, SCR_WIDTH, SCR_HEIGHT);

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

        if(showWindow)
        {
            ImGui::ShowPerfomanceMetrics();

            ImGui::Begin("Camera");
            ImGui::DragFloat3("Front", camera.front.value_ptr(), 0.1f, -10.0f, 10.0f);
            ImGui::DragFloat3("Position", camera.e.value_ptr(), 0.1f, -10.0f, 10.0f);
            ImGui::DragFloat("Yaw angle", &camera.yaw, 0.1f, -90.0f, 90.0f);
            ImGui::DragFloat("Pitch angle", &camera.pitch, 0.1f, -90.0f, 90.0f);
            ImGui::End();

            ImGui::Begin("1st Light");
            ImGui::DragFloat3("Position", light1.value_ptr(), 0.1f, -20.0f, 20.0f);
            ImGui::End();

            ImGui::Begin("2nd Light");
            ImGui::DragFloat3("Position", light2.value_ptr(), 0.1f, -20.0f, 20.0f);
            ImGui::End();

            ImGui::Begin("Bounces");
            ImGui::InputInt("Bounces", &depth);
            ImGui::End();
        }

        ImGui::Render();

        canvas.renderTexture();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_handler.window);

        //primary_total = 0;
        //secondary_total = 0;
        //for (int i = 0; i < N; i++) { primary_total += primary_rays[i]; }
        //for (int i = 0; i < N*7; i++) { secondary_total += secondary_rays[i]; }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    scene.destroy();

    //std::cout << "Number of traced primary rays: " << "\n" << "\t" << primary_total << "\n";
    //std::cout << "Number of traced secondary rays: " << "\n" << "\t" << secondary_total << "\n";

    //cudaFree(primary_rays);
    //cudaFree(secondary_rays);
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


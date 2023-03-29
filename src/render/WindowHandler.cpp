#include "WindowHandler.h"

#include <iostream>

WindowHandler::WindowHandler(const int& width, const int& height, const char* title)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, title, NULL, NULL);

    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwSetFramebufferSizeCallback(window, WindowHandler::framebufferCallback);
    glfwSetWindowUserPointer(window, static_cast<void *>(this));
}

WindowHandler::~WindowHandler()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

void WindowHandler::mark_as_current_context()
{
    glfwMakeContextCurrent(window);
}

void WindowHandler::setViewportSize(const int& width, const int& height)
{
    glViewport(0, 0, width, height);
}

void WindowHandler::framebufferCallback(GLFWwindow* window, int width, int height)
{
     WindowHandler* instance = static_cast<WindowHandler*>(glfwGetWindowUserPointer(window));
     if (instance)
         instance->setViewportSize(width, height);
}

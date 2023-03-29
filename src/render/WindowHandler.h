#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class WindowHandler
{
public:
	GLFWwindow* window;

	WindowHandler(const int& width, const int& height, const char* title);
	~WindowHandler();
	void mark_as_current_context();	
private:
	void setViewportSize(const int& width, const int& height);
	static void framebufferCallback(GLFWwindow* window, int width, int height);
};


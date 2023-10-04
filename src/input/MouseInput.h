#pragma once
#include <imgui_impl_glfw.h>
#include <GLFW/glfw3.h>

class MouseInput
{
public:
	MouseInput() = default;
	MouseInput(const float& sensitivity, const int& width, const int& height, float& pitch, float& yaw);
	static void setMouseInput(GLFWwindow* window);
	void pointerMode(GLFWwindow* window, bool state);
private:
	static MouseInput* instance;
	float _sensitivity;
	float& _pitch;
	float& _yaw;
	float _lastX;
	float _lastY;
	bool  _firstMouse;
	static float PITCH_BOUND;

	void setRotationOffset(const float& xpos, const float& ypos);
	static void callback(GLFWwindow* window, double xpos, double ypos);
};


#include "MouseInput.h"

MouseInput* MouseInput::instance = nullptr;

MouseInput::MouseInput(const float& sensitivity, const int& width, const int& height, float& pitch, float& yaw) :
	_sensitivity(sensitivity), _pitch(pitch), _yaw(yaw), _firstMouse(true)
{
	_lastX = static_cast<float>(width) / 2.0f;
	_lastY = static_cast<float>(height) / 2.0f;
	MouseInput::instance = this;
}

void MouseInput::setMouseInput(GLFWwindow* window)
{
	glfwSetCursorPosCallback(window, NULL);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void MouseInput::pointerMode(GLFWwindow* window, bool state)
{
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED - (2 * static_cast<int>(state)));
	if (state == true)
	{
		glfwSetCursorPosCallback(window, ImGui_ImplGlfw_CursorPosCallback);
		_firstMouse = true;
	}
	else
		glfwSetCursorPosCallback(window, MouseInput::callback);
}

void MouseInput::setRotationOffset(const float& xpos, const float& ypos)
{
	if (_firstMouse)
	{
		_lastX = xpos;
		_lastY = ypos;
		_firstMouse = false;
	}

	float xoffset = xpos - _lastX;
	float yoffset = ypos - _lastY;

	_lastX = xpos;
	_lastY = ypos;

	xoffset *= _sensitivity;
	yoffset *= _sensitivity;

	_yaw += xoffset;
	_pitch += yoffset;
}

void MouseInput::callback(GLFWwindow* window, double xpos, double ypos)
{
	xpos = static_cast<float>(xpos);
	ypos = static_cast<float>(ypos);

	MouseInput::instance->setRotationOffset(xpos, ypos);
}

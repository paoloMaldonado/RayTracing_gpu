#include "KeyInput.h"

KeyInput* KeyInput::instance = nullptr;



KeyInput::KeyInput(const std::vector<int>& keysToMonitor, const std::vector<int>& modsToMonitor)
{
	for (const auto& key : keysToMonitor)
	{
		keyMap[key] = false;
	}

	for (const auto& mod : modsToMonitor)
	{
		modsMap[mod] = false;
	}

	KeyInput::instance = this;
}

void KeyInput::setKeyboardInput(GLFWwindow* window)
{
	glfwSetKeyCallback(window, KeyInput::callback);
	glfwSetInputMode(window, GLFW_MOD_ALT, GLFW_TRUE);
}

bool KeyInput::isKeyDown(const int& key)
{
	return keyMap[key];
}

bool KeyInput::isModPressed(const int& key, const int& mod)
{
	return modsMap[mod+key];
}

void KeyInput::setKeyDown(const int& key, bool isDown)
{
	auto it = keyMap.find(key);
	if(it != keyMap.end())
		keyMap[key] = isDown;
}

void KeyInput::setKeyModPressed(const int& key, const int& mod, const int& action)
{
	if (action != GLFW_PRESS)
		return;
	auto it = modsMap.find(mod+key);
	if (it != modsMap.end())
		modsMap[mod+key] = !modsMap[mod+key];
}

void KeyInput::callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	instance->setKeyDown(key, action != GLFW_RELEASE);
	instance->setKeyModPressed(key, mods, action);
}

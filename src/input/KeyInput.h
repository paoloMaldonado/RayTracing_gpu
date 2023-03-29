#pragma once
#include <GLFW/glfw3.h>
#include <map>
#include <vector>

#define GLFW_MOD_ALT_X 430	// 342+88

class KeyInput
{
public:
	KeyInput() = default;
	KeyInput(const std::vector<int>& keysToMonitor, const std::vector<int>& modsToMonitor);
	static void setKeyboardInput(GLFWwindow* window);
	bool isKeyDown(const int& key);
	bool isModPressed(const int& key, const int& mod);
private:
	std::map<int, bool> keyMap;
	std::map<int, bool> modsMap;
	static KeyInput* instance;
	void setKeyDown(const int& key, bool isDown);
	void setKeyModPressed(const int& key, const int& mod, const int& action);
	static void callback(GLFWwindow* window, int key, int scancode, int action, int mods);
};


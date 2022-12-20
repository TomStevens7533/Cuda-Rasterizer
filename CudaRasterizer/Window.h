#pragma once

struct GLFWwindow;
class Window
{
public:
	Window(int width, int height);
	~Window();
	void InitWindow();
	void SwapWindow();
	GLFWwindow* GetWindow();
private:
	void ShowFPS();
private:
	int m_width{}, m_height{};
	GLFWwindow* m_Window;
	int frame{};
	double lastTime{};
};


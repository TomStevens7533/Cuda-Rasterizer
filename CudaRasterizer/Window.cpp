#include "Window.h"
#include <sstream>
#include <iostream>
#include <glfw3.h>
Window::Window(int width, int height) : m_width{ width }, m_height{ height }
{

}

Window::~Window()
{
	glfwDestroyWindow(m_Window);
	glfwTerminate();
}

void Window::InitWindow()
{
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	if (!glfwInit())
	{
		// Initialization failed
		std::cout << "glfw failed initialization!\n";
	}
	m_Window = glfwCreateWindow(m_width, m_height, "Very cool rasterizer ow ye", NULL, NULL);
	if (!m_Window)
	{
		glfwTerminate();
		// Window or OpenGL context creation failed
	}
	glfwMakeContextCurrent(m_Window);

	//GLFW CALLBACKS
	glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int Key, int scancode, int action, int mods)
		{
			//Set GLFW to close next frame
			if (Key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {

			}
		}
	);
	glfwSetErrorCallback([](int error, const char* description) {
		std::cerr << "GLFW error detected: " << description << std::endl;
		}
	); 
	glfwSwapInterval(0);
}

void Window::SwapWindow()
{
	glfwSwapBuffers(m_Window);
	glfwPollEvents();
	ShowFPS();
}

GLFWwindow* Window::GetWindow()
{
	return m_Window;
}

void Window::ShowFPS()
{
	double currentTime = glfwGetTime();
	double delta = currentTime - lastTime;
	frame++;
	if (delta >= 1.0) { // If last cout was more than 1 sec ago

		double fps = double(frame) / delta;

		std::stringstream ss;
		ss << "Soy de meigd" << " [" << fps << " FPS]";

		glfwSetWindowTitle(m_Window, ss.str().c_str());

		frame = 0;
		lastTime = currentTime;
	}
}


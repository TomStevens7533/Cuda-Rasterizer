#include <iostream>
#include <GLFW/glfw3.h>
#include <gl/GL.h>
#include <glm/glm.hpp>

void InitGLFW();

int main() {
	InitGLFW();
	std::cout << "Hello world\n";
}


void InitGLFW()
{
	if (!glfwInit())
	{
		// Initialization failed
		std::cout << "glfw failed initialization!\n";
	}
}
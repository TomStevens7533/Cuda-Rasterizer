#include "Application.h"
#include <vector>
#include "Structs.h"
#include "Renderer.h"

Application* Application::s_Instance = nullptr;

Application::Application(int width, int height) : m_Window{width, height}
{
	s_Instance = this;
	m_Window.InitWindow();
}

Application::~Application()
{
	
}



void Application::Start()
{

}

void Application::Update()
{
	std::vector<Input_Triangle> triangleVec;
	Input_Triangle tr;
	tr.vertices[0] = glm::vec3{ 0.f, 0.5f,-1.f };
	tr.vertices[1] = glm::vec3{ -.5f, -0.5f,-1.f };
	tr.vertices[2] = glm::vec3{ 0.5f, -0.5f,-1.f };
	triangleVec.push_back(tr);
	glm::mat4x4 trans = glm::translate(glm::mat4(1), glm::vec3{0,0, -1});
	Renderer::Submit(triangleVec.data(), triangleVec.size(), trans);
}

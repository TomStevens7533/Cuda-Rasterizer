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
	tr.vertices[0] = glm::vec3{ 0.f, 2.f, 0.f };
	tr.vertices[1] = glm::vec3{ -1.f, 0.f, 0.f };
	tr.vertices[2] = glm::vec3{ 1.f, 0.f,  0.f };
	Input_Triangle tr1;
	tr1.vertices[0] = glm::vec3{ 0.f, 2.f, 2.f };
	tr1.vertices[1] = glm::vec3{ -1.f, 0.f, 2.f };
	tr1.vertices[2] = glm::vec3{ 4.f, 0.f,  2.f };

	triangleVec.push_back(tr);
	triangleVec.push_back(tr1);


	glm::mat4x4 trans = glm::translate(glm::mat4(1), glm::vec3{0,0, 0});
	Renderer::Submit(triangleVec.data(), triangleVec.size(), trans);
}

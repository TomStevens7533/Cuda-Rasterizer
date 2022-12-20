#include "Application.h"
#include <vector>
#include "Structs.h"
#include "Renderer.h"


Application::Application()
{

}

Application::~Application()
{

}

void Application::Start()
{

}

void Application::Update()
{
	std::vector<Triangle> triangleVec;
	Triangle tr;
	tr.vertices[0] = glm::vec3{ 0.f, 0.5f,-1.f };
	tr.vertices[1] = glm::vec3{ -.5f, -0.5f,-1.f };
	tr.vertices[2] = glm::vec3{ 0.5f, -0.5f,-1.f };
	triangleVec.push_back(tr);
	Renderer::Submit(triangleVec.data(), triangleVec.size(), glm::mat4());
}

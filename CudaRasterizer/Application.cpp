#include "Application.h"
#include <vector>
#include "Structs.h"
#include "Renderer.h"
#include <array>
#include "ChunkBuilder.h"
#include "Block.h"
#include "Camera.h"



Application* Application::s_Instance = nullptr;

Application::Application(int width, int height, Camera* pCam) : m_Window{width, height},
chk{ pCam->GetPosition(), 40 }
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

void Application::Update(Camera* pCam)
{
	chk.SwapBuffers(pCam->GetPosition(), 40);
	//Set on seperate thread
	//chk.TraverseSVO(pCam->GetPosition(), 20);

	//std::vector < glm::vec3 > vertexBuff;
	//std::vector < int > indexBuff;
	//
	//vertexBuff.push_back(glm::vec3{ -3.f,3.f,-2.f });
	//vertexBuff.push_back(glm::vec3{ 0.f,3.f,-2.f });
	//vertexBuff.push_back(glm::vec3{ 3.f,3.f,-2.f });
	//vertexBuff.push_back(glm::vec3{ -3.f,0.f,-2.f });
	//vertexBuff.push_back(glm::vec3{ 0.f,0.f,-2.f });
	//vertexBuff.push_back(glm::vec3{ 3.f,0.f,-2.f });
	//vertexBuff.push_back(glm::vec3{ -3.f,-3.f,-2.f });
	//vertexBuff.push_back(glm::vec3{ 0.f,-3.f,-2.f });
	//vertexBuff.push_back(glm::vec3{ 3.f,-3.f,-2.f });
	//
	//
	//
	//indexBuff.push_back(0);
	//indexBuff.push_back(3);
	//indexBuff.push_back(1);
	//
	//
	//indexBuff.push_back(3);
	//indexBuff.push_back(4);
	//indexBuff.push_back(1);
	//
	//indexBuff.push_back(1);
	//indexBuff.push_back(4);
	//indexBuff.push_back(2);
	////
	//indexBuff.push_back(4);
	//indexBuff.push_back(5);
	//indexBuff.push_back(2);
	////
	//indexBuff.push_back(3);
	//indexBuff.push_back(6);
	//indexBuff.push_back(4);
	////
	//indexBuff.push_back(6);
	//indexBuff.push_back(7);
	//indexBuff.push_back(4);
	////
	//indexBuff.push_back(4);
	//indexBuff.push_back(7);
	//indexBuff.push_back(5);
	//
	//indexBuff.push_back(7);
	//indexBuff.push_back(8);
	//indexBuff.push_back(5);
	//Chunk chk;





	Chunk blx;
	glm::mat4x4 trans = glm::translate(glm::mat4(1), glm::vec3{0,0, 0});
	Renderer::Submit(chk.GetVertices().data(), chk.GetVertices().size(), chk.GetIndices().data(), chk.GetIndices().size(), trans);
}

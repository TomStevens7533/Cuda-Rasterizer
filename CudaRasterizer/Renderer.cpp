#include "Renderer.h"
#include "CudaKernel.h"
#include "Camera.h"

SceneData Renderer::m_CurrData;

void Renderer::BeginScene(const SceneData sceneData)
{
	m_CurrData = sceneData;
}

void Renderer::Submit(const Input_Triangle* pTriangles, int triangleAmount, const glm::mat4& transform /*= glm::mat4(1.0f)*/)
{
	glm::mat4x4 matrix = (m_CurrData.pCamera->GetViewProjectionMatrix());
	cudaRasterizeCore(m_CurrData.PBOpos, m_CurrData.resolution, pTriangles, triangleAmount,  matrix * transform);
}





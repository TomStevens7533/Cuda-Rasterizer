#pragma once
#include "Structs.h"

class Camera;
struct uchar4;
struct SceneData
{
	Camera* pCamera;
	glm::vec2 resolution;
	uchar4* PBOpos;
};

class Renderer
{
public:
	static void BeginScene(const SceneData sceneData);
	static void Submit(const glm::vec3* pVertexBuffer, int vertexBufferAmount, int* pIndexBuffer, int indexBufferAmount, const glm::mat4& transform = glm::mat4(1.0f));
private:
	static SceneData m_CurrData;
};

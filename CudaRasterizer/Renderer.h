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
	static void Submit(const Input_Triangle* pTriangles, int triangleAmount, const glm::mat4& transform = glm::mat4(1.0f));
private:
	static SceneData m_CurrData;
};

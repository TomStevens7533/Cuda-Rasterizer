#pragma once
#include "ext/matrix_transform.hpp"

struct Input_Triangle
{
	glm::vec3 worldSpaceCoords[3];
	glm::vec4 viewspaceCoords[3];
	glm::vec4 NDC[3];
	glm::vec3 Screenspace[3];
	glm::vec3 Normal;
	glm::vec3 Color;
};
struct AABB {
	glm::vec3 min;
	glm::vec3 max;
};
struct cudaMat3 {
	glm::vec3 x;
	glm::vec3 y;
	glm::vec3 z;
};

struct light {
	glm::vec3 position;
	glm::vec3 diffColor;
	glm::vec3 specColor;
	int specExp;
	glm::vec3 ambColor;
};
struct cudaMat4 {
	glm::vec4 x;
	glm::vec4 y;
	glm::vec4 z;
	glm::vec4 w;
};


struct fragment {
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec3 position;
	glm::vec3 cameraSpacePosition;
	float depth;
	bool isEmpty;
};


#pragma once
#include "ext/matrix_transform.hpp"

struct Input_Triangle
{
	glm::vec3 vertices[3];
};
struct Output_Triangle
{
	glm::vec3 vertices[3];
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
	bool isEmpty;
};


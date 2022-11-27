#pragma once
#include "ext/matrix_transform.hpp"
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

struct triangle {
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 p2;
	glm::vec3 c0;
	glm::vec3 c1;
	glm::vec3 c2;
	glm::vec3 n0;
	glm::vec3 n1;
	glm::vec3 n2;
	glm::vec3 flatNormal;
};

struct fragment {
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec3 position;
	glm::vec3 cameraSpacePosition;
	bool isEmpty;
	bool isFlat;
	float coverage;
};


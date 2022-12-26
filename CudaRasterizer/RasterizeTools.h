#pragma once
#include "Structs.h"

__device__ AABB getAABBForTriangle(const Output_Triangle tri)
{
	AABB aabb;
	aabb.min = glm::vec3(
		min(min(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x),
		min(min(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y),
		min(min(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z));
	aabb.max = glm::vec3(
		max(max(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x),
		max(max(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y),
		max(max(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z));
	return aabb;
}
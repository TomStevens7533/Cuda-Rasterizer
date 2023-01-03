#pragma once
#include "Structs.h"

__device__ AABB getBBForTriangle(const Input_Triangle tri)
{
	AABB aabb;
	aabb.min = glm::vec3(
		min(min(tri.Screenspace[0].x, tri.Screenspace[1].x), tri.Screenspace[2].x),
		min(min(tri.Screenspace[0].y, tri.Screenspace[1].y), tri.Screenspace[2].y),
		min(min(tri.Screenspace[0].z, tri.Screenspace[1].z), tri.Screenspace[2].z));
	aabb.max = glm::vec3(										 
		max(max(tri.Screenspace[0].x, tri.Screenspace[1].x), tri.Screenspace[2].x),
		max(max(tri.Screenspace[0].y, tri.Screenspace[1].y), tri.Screenspace[2].y),
		max(max(tri.Screenspace[0].z, tri.Screenspace[1].z), tri.Screenspace[2].z));
	return aabb;
}
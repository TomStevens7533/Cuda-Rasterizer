#pragma once
#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <glm.hpp>
#include "Structs.h"



void kernelCleanup();
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, const glm::vec3* pVertexBuffer, int vertexAnmount, const int* pIndexBuffer, int indexAmount, glm::mat4 worldviewprojMat);
void InitializeBuffers(glm::vec2 resolution);
void ClearImage(glm::vec2 resolution);

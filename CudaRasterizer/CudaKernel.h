#pragma once
#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <glm.hpp>
#include "Structs.h"


void kernelCleanup();
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, Triangle* TrArray, int TriangleSize);
void InitializeBuffers(glm::vec2 resolution);

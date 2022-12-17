#pragma once
#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <glm.hpp>
#include "Structs.h"


void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, Triangle* TrArray, int TriangleSize, glm::mat4 glmViewTransform, glm::mat4 glmProjectionTransform, glm::mat4 glmMVtransform);

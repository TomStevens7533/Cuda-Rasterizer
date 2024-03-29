#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <glm.hpp>


struct light {
	glm::vec3 position;
	glm::vec3 diffColor;
	glm::vec3 specColor;
	int specExp;
	glm::vec3 ambColor;
};

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, glm::mat4 glmViewTransform, glm::mat4 glmProjectionTransform, glm::mat4 glmMVtransform, light Light, int isFlatShading, int isMeshView);

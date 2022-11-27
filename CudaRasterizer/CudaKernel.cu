
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "CudaKernel.h"

#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


glm::vec3* framebuffer;
fragment* depthbuffer;

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ 
void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {

        glm::vec3 color;
        color.x = image[index].x * 255.0;
        color.y = image[index].y * 255.0;
        color.z = image[index].z * 255.0;

        if (color.x > 255) {
            color.x = 255;
        }

        if (color.y > 255) {
            color.y = 255;
        }

        if (color.z > 255) {
            color.z = 255;
        }

        // Each thread writes one pixel location in the texture (textel)
        PBOpos[index].w = 0;
        PBOpos[index].x = color.x;
        PBOpos[index].y = color.y;
        PBOpos[index].z = color.z;
    }
}
//Writes fragment colors to the framebuffer
__global__ 
void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {
        framebuffer[index] = glm::vec3{255.f, 0.f, 0.f};
    }
}
// Wrapper for the __global__ call that sets up the kernel calls and memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, glm::mat4 glmViewTransform, glm::mat4 glmProjectionTransform, glm::mat4 glmMVtransform, light Light, int isFlatShading, int isMeshView) {

    // set up thread configuration
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

    //set up framebuffer
    framebuffer = NULL;
    int success = cudaMalloc((void**)&framebuffer, (int)resolution.x * (int)resolution.y * sizeof(glm::vec3));

    //set up depthbuffer
    depthbuffer = NULL;
    success = cudaMalloc((void**)&depthbuffer, (int)resolution.x * (int)resolution.y * sizeof(fragment));

    render << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, depthbuffer, framebuffer);
    sendImageToPBO << <fullBlocksPerGrid, threadsPerBlock >> > (PBOpos, resolution, framebuffer);

    cudaDeviceSynchronize();


    kernelCleanup();
}


void kernelCleanup() {
    cudaFree(framebuffer);
    cudaFree(depthbuffer);

}

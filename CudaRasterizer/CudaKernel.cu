﻿
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "CudaKernel.h"

#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>


__device__ Triangle* trDeviceArray;
__device__ glm::vec3* framebuffer;
__device__ fragment* depthbuffer;

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        // exit(EXIT_FAILURE); 
    }
}



//How to call only other device fucntions+
glm::vec2 ConvertToScreenspace(const glm::vec3 ndcPos, glm::vec2 resolution) {
   
    glm::vec2 screenspacePoint = glm::vec2{ ((ndcPos.x + 1) * 0.5f),  ((1 - ndcPos.y) * 0.5f) * resolution.y };
    return screenspacePoint;   
}
//Find bounding box
 __device__ void getBoundingBoxForTriangle(Triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint) {
    minpoint = glm::vec3(min(min(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x),
        min(min(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y),
        min(min(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z));

    maxpoint = glm::vec3(max(max(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x),
        max(max(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y),
        max(max(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z));
}
__host__ __device__ float calculateSignedArea(Triangle tri) {
    return 0.5 * ((tri.vertices[2].x - tri.vertices[0].x) * (tri.vertices[1].y - tri.vertices[0].y)
        - (tri.vertices[1].x - tri.vertices[0].x) * (tri.vertices[2].y - tri.vertices[0].y));
}
__device__ float Cross(glm::vec2 lhs, glm::vec2 rhs) {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}
__device__ bool IsPixelInTriangle(Triangle tri, const glm::vec2 point, float* pWeight, glm::vec2 resolution) {

    glm::vec2 ScreenspaceA = glm::vec2{ ((tri.vertices[0].x + 1) * 0.5f),  ((1 - tri.vertices[0].y) * 0.5f) * resolution.y };
    glm::vec2 ScreenspaceB = glm::vec2{ ((tri.vertices[1].x + 1) * 0.5f),  ((1 - tri.vertices[1].y) * 0.5f) * resolution.y };
    glm::vec2 ScreenspaceC = glm::vec2{ ((tri.vertices[2].x + 1) * 0.5f),  ((1 - tri.vertices[2].y) * 0.5f) * resolution.y };

    glm::vec2 edgeA = glm::vec2(ScreenspaceB - ScreenspaceA);
    glm::vec2 edgeB = glm::vec2(ScreenspaceC - ScreenspaceB);
    glm::vec2 edgeC = glm::vec2(ScreenspaceA - ScreenspaceC);

    glm::vec2 pixelToVertex{ ScreenspaceA - point };

    bool IsIntersection = true;
    float weight2 = Cross(edgeA, pixelToVertex);
    if (weight2 < 0.f) { //no intersection
        return false;
    }
    pixelToVertex = ScreenspaceB - point;
    float weight1 = Cross(edgeB, pixelToVertex);
    if (weight1 < 0.f) { //no intersection
        return false;
    
    }
    pixelToVertex = ScreenspaceC - point;
    float weight0 = Cross(edgeC, pixelToVertex);
    if (weight0 < 0.f) { //no intersection
        return false;
    }
    float totalSurface = Cross(edgeA, edgeC);
    pWeight[0] = weight2 / totalSurface;
    pWeight[1] = weight1 / totalSurface;
    pWeight[2] = weight0 / totalSurface;
    printf("Lol de meigd");
    return true;
}
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
__global__
void RasterizeKernel(Triangle* primitives, int triangleSize, glm::vec2 resolution) {
    int triangleIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (triangleIdx < triangleSize) {
        Triangle currTriangle = primitives[triangleIdx];
        //ConvertToScreenspace(currTriangle, resolution);

        glm::vec3 Min, Max;
        getBoundingBoxForTriangle(currTriangle, Min, Max);
        float pixelWidth = 1.0f / (float)resolution.x;
        float pixelHeight = 1.0f / (float)resolution.y;
        float weights[3]{};
        float halfResoX = 0.5f * (float)resolution.x;
        float halfResoY = 0.5f * (float)resolution.y;
        //TODO Add Bounding Box
        for (float i = 0; i < (Max.x - Min.x) / pixelWidth + 1.0f; i += 1.f)
        {
            for (float j = 0; j < (Max.y - Min.y) / pixelHeight + 1.0f; j += 1.f)
            {
              
               glm::vec2 pixelPos = glm::vec2(Min.x + i * pixelWidth, Min.y + j * pixelHeight);

               fragment frag;
               frag.isEmpty = false;
               if (IsPixelInTriangle(currTriangle, pixelPos,weights, resolution)) { //In in triangle

                   int x, y, pixelIndex;
       

                   x = pixelPos.x * halfResoX + halfResoX;
                   y = pixelPos.y * halfResoY + halfResoY;
                   if (x < 0 || y < 0 || x > resolution.x || y > resolution.y) continue;
                   pixelIndex = x + y * resolution.x;


                   frag.color = glm::vec3{ 0.f,0.f,0.f };
                   depthbuffer[pixelIndex] = frag;
               
               }
               else {
               }
               //if (!isBarycentricCoordInBounds(pixelBarycentricPos))
               //    continue;

               //Pixel is InTriangle

            }
        }
    }

}

//Writes fragment colors to the framebuffer
__global__ 
void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {

        framebuffer[index] = depthbuffer[index].color;
        //framebuffer[index] = glm::vec3{255.f, 0.f, 0.f};

    }
}

void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, Triangle* TrArray, int TriangleSize, glm::mat4 glmViewTransform, glm::mat4 glmProjectionTransform, glm::mat4 glmMVtransform) {
    //Init buffers
    //set up framebuffer
    checkCUDAError("Setup failed");

    framebuffer = NULL;
    cudaError_t err = cudaMalloc((void**)&framebuffer, (int)resolution.x * (int)resolution.y * sizeof(glm::vec3));
    checkCUDAError("Init framebuffer failed");

    //set up depthbuffer
    depthbuffer = NULL;
    err = cudaMalloc((void**)&depthbuffer, (int)resolution.x * (int)resolution.y * sizeof(fragment));
    checkCUDAError("Init depthbuffer failed");


    //Move Host memory To Device; Host == CPU && Device == GPU
    trDeviceArray = NULL;

    // Allocate Unified Memory – accessible from CPU or GPU
    err = cudaMallocManaged((void**)&trDeviceArray, TriangleSize * sizeof(Triangle));
    err = cudaMemcpy(trDeviceArray, TrArray, TriangleSize * sizeof(Triangle), cudaMemcpyHostToDevice);
    checkCUDAError("Copying Triangle data failed");


    // set up thread configuration
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

    RasterizeKernel <<<fullBlocksPerGrid, threadsPerBlock>>>(trDeviceArray, TriangleSize, resolution);
 
    cudaDeviceSynchronize();

    //Rasterization

    //Primitive assembly


    render << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, depthbuffer, framebuffer);
    sendImageToPBO << <fullBlocksPerGrid, threadsPerBlock >> > (PBOpos, resolution, framebuffer);

    cudaDeviceSynchronize();


    cudaFree(trDeviceArray);
    cudaFree(framebuffer);
    cudaFree(depthbuffer);
}

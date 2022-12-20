
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <thrust/random.h>
#include "CudaKernel.h"
#include <atomic>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>

__device__ int* depthBufferLock;
__device__ Triangle* trDeviceArray;
__device__ glm::vec3* framebuffer;
fragment* depthbuffer;

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
            exit(EXIT_FAILURE); 
    }
}

//fast initializor for int array
__global__ void initiateArray(int* array, int val, int num)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < num)
    {
        array[index] = val;
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

    glm::vec2 ScreenspaceA = glm::vec2{ ((tri.vertices[0].x + 1) * 0.5f) * resolution.x ,  ((1 - tri.vertices[0].y) * 0.5f) * resolution.y };
    glm::vec2 ScreenspaceB = glm::vec2{ ((tri.vertices[1].x + 1) * 0.5f) * resolution.x,  ((1 - tri.vertices[1].y) * 0.5f) * resolution.y };
    glm::vec2 ScreenspaceC = glm::vec2{ ((tri.vertices[2].x + 1) * 0.5f) * resolution.x,  ((1 - tri.vertices[2].y) * 0.5f) * resolution.y };

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
void RasterizeKernel(Triangle* primitives, int triangleSize, glm::vec2 resolution, fragment* depthBuffer) {
    int triangleIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (triangleIdx < triangleSize) {
        Triangle currTriangle = primitives[triangleIdx];

        glm::vec3 Min, Max;
        getBoundingBoxForTriangle(currTriangle, Min, Max);
        float pixelWidth = 1.0f / (float)resolution.x;
        float pixelHeight = 1.0f / (float)resolution.y;
        float weights[3]{};

        //TODO Add Bounding Box
        for (int i = 0; i < resolution.x; i += 1)
        {
            for (int j = 0; j < resolution.y; j += 1)
            {
              
               glm::vec2 pixelPos = glm::vec2(i, j);

               fragment frag;
               frag.isEmpty = false;
               if (IsPixelInTriangle(currTriangle, pixelPos,weights, resolution)) { //In in triangle

                   int pixelIdx = i + (j * resolution.x);

                   bool shouldWait = true;

                   frag.color = glm::vec3{ 255.f, 0.f, 0.f };
                   depthBuffer[pixelIdx] = frag;
                 
                  

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
void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer, fragment* depthBuffer) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {

        framebuffer[index] = depthBuffer[index].color;
        //framebuffer[index] = glm::vec3{ 255.f, 0.f, 0.f };

        //if (fr.x == 255.f)
        //{
        //    printf("de meigd");
        //}

    }
}
void InitializeBuffers(glm::vec2 resolution) {

    framebuffer = NULL;
    cudaError_t err = cudaMalloc((void**)&framebuffer, (int)resolution.x * (int)resolution.y * sizeof(glm::vec3));
    checkCUDAError("Init framebuffer failed");

    //set up depthbuffer
    depthbuffer = NULL;
    err = cudaMalloc((void**)&depthbuffer, (int)resolution.x * (int)resolution.y * sizeof(fragment));
    checkCUDAError("Init depthbuffer failed");

    int depthBufferLockSize = resolution.x * resolution.y;
    depthBufferLock = NULL;
    cudaMalloc((void**)&depthBufferLock, depthBufferLockSize * sizeof(int));
    initiateArray << <dim3(ceil((float)depthBufferLockSize / ((float)512))), dim3(512) >> > (depthBufferLock, 0, depthBufferLockSize);
    checkCUDAError("Init depthbufferLock failed");


}

void kernelCleanup()
{
    cudaFree(trDeviceArray);
    cudaFree(framebuffer);
    cudaFree(depthbuffer);
    cudaFree(depthBufferLock);
}


void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, Triangle* TrArray, int TriangleSize) {
    //Init buffers
    //set up framebuffer
    checkCUDAError("Setup failed");

    //Move Host memory To Device; Host == CPU && Device == GPU
    trDeviceArray = NULL;
    cudaError_t err;
    // Allocate Unified Memory – accessible from CPU or GPU
    err = cudaMallocManaged((void**)&trDeviceArray, TriangleSize * sizeof(Triangle));
    err = cudaMemcpy(trDeviceArray, TrArray, TriangleSize * sizeof(Triangle), cudaMemcpyHostToDevice);
    checkCUDAError("Copying Triangle data failed");

    // set up thread configuration
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

    RasterizeKernel <<<fullBlocksPerGrid, threadsPerBlock>>>(trDeviceArray, TriangleSize, resolution, depthbuffer);
    cudaDeviceSynchronize();

    checkCUDAError("Rasterization failed");

    //Rasterization

    //Primitive assembly


   render << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, depthbuffer, framebuffer, depthbuffer);
   checkCUDAError("Render failed");
   sendImageToPBO << <fullBlocksPerGrid, threadsPerBlock >> > (PBOpos, resolution, framebuffer);
   checkCUDAError("Sending To PBO failed");


    cudaDeviceSynchronize();


}

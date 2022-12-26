﻿
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
#include "RasterizeTools.h"

struct Tile {
    int m_TriangleIndexes[1000]; //Indexes of triangles that are in this Tile
    int m_TriangleAmount{ 0 };
};

__device__ int* depthBufferLock;

__device__  int* dev_pIndexBuffer = nullptr;
__device__  glm::vec3* dev_pVertexBuffer = nullptr;
__device__  glm::vec4* dev_pOutputVertexBuffer = nullptr;

__device__  Input_Triangle* dev_pTriangleBuffer = nullptr;

__device__ glm::vec3* framebuffer;
Tile* tileBuffer = NULL;
int* dev_tileMutex = NULL;
fragment* depthbuffer;
glm::mat4x4* MV;

#define TILE_H_AMOUNT 32
#define TILE_W_AMOUNT 32
int numTiles = (TILE_W_AMOUNT + 1) * (TILE_H_AMOUNT + 1);


int m_CurrentBufferPrimitiveAmount{0};

void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
            exit(EXIT_FAILURE); 
    }
}
void kernelCleanup()
{
    cudaFree(dev_pTriangleBuffer);
    cudaFree(dev_pIndexBuffer);
    cudaFree(dev_pVertexBuffer);

    cudaFree(framebuffer);
    cudaFree(depthbuffer);
    cudaFree(depthBufferLock);
    cudaFree(tileBuffer);
    cudaFree(dev_tileMutex);

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
__host__ __device__ float calculateSignedArea(Input_Triangle tri) {
    return 0.5 * ((tri.vertices[2].x - tri.vertices[0].x) * (tri.vertices[1].y - tri.vertices[0].y)
        - (tri.vertices[1].x - tri.vertices[0].x) * (tri.vertices[2].y - tri.vertices[0].y));
}
__device__ float Cross(glm::vec2 lhs, glm::vec2 rhs) {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}
__device__ bool IsPixelInTriangle(Output_Triangle tri, const glm::vec2 point, float* pWeight, glm::vec2 resolution) {
    //Maybe positive values get passed this too

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
void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, fragment* framebuffer) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x <= resolution.x && y <= resolution.y) {

        glm::vec3 color = framebuffer[index].color;

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
__device__ uint32_t clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}
//LOOK: finds the axis aligned bounding box for a given triangle
__host__ __device__ void getAABBForTriangle(Output_Triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint) {
    minpoint = glm::vec3(min(min(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x),
        min(min(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y),
        min(min(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z));
    maxpoint = glm::vec3(max(max(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x),
        max(max(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y),
        max(max(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z));
}
__global__ void SortTrianglesInCorrectTile(Input_Triangle* primitives, int triangleSize, glm::vec2 resolution, int strideX, int strideY, int* dev_tilemutex, Tile* dev_tilebuffer) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < triangleSize) {
        Input_Triangle input = primitives[index];
        Output_Triangle output;

        //Transform vertices
        for (size_t i = 0; i < 3; i++)
        {

            //Convert to NDC coordinates
            glm::vec4 NDC = glm::vec4(input.vertices[i].x / input.vertices[i].w, input.vertices[i].y / input.vertices[i].w, input.vertices[i].z / input.vertices[i].w, input.vertices[i].w);
            //Convert To Screenspace
            glm::vec3 screenSpace;

            screenSpace.x = ((NDC.x + 1) / 2) * resolution.x;
            screenSpace.y = ((1 - NDC.y) / 2) * resolution.y;
            screenSpace.z = ((NDC.z + 1.0f) * 0.5f);
            output.vertices[i] = screenSpace;


        }
        AABB boundingBox = getAABBForTriangle(output);
        int tileidminX = glm::floor(boundingBox.min.x / strideX);
        int tileidmaxX = glm::ceil(boundingBox.max.x /  strideX);
        int tileidminY = glm::floor(boundingBox.min.y / strideY);
        int tileidmaxY = glm::ceil(boundingBox.max.y /  strideY);

        int tilesX = (int)glm::ceil(double(resolution.x) / double(strideX));

        for (int i = tileidminY; i < tileidmaxY; i++)
        {
            for (int j = tileidminX; j < tileidmaxX; j++)
            {
                int tileID = j * i;
                bool isSet = false;
                do
                {

                    isSet = atomicCAS(&dev_tilemutex[tileID], 0, 1) == 0;
                    if (isSet)
                    {
                        int t = dev_tilebuffer[tileID].m_TriangleAmount;
                        dev_tilebuffer[tileID].m_TriangleIndexes[t] = index;
                        dev_tilebuffer[tileID].m_TriangleAmount = t + 1;
                        dev_tilemutex[tileID] = 0;
                    }

                   //isSet = (atomicCAS(&dev_tileMutex[tileID], 0, 1) == 0);
                   //if (isSet)
                   //{
                   //    // Critical section goes here.
                   //     //if it is afterward, a deadlock will occur.
                   //    
                   //    
                   //    
                   //
                   //    atomicExch(&dev_tileMutex[tileID], 0);
                   //}
                } while (!isSet);

            }
        }
    }
}
//void RasterizeKernel(Input_Triangle* primitives, int triangleSize, glm::vec2 resolution, fragment* depthBuffer) {
//   
//        Input_Triangle currTriangle = primitives[0];
//        Output_Triangle outputTriangle;
//        glm::vec2 screenspaceVec[3];
//
//        uint16_t min_x = UINT16_MAX;
//        uint16_t max_x = 0;
//        uint16_t min_y = UINT16_MAX;
//        uint16_t max_y = 0;
//
//      
//        float blockDimWidth = resolution.x / blockDim.x;
//        float blockDimHeight = resolution.y / blockDim.x;
//
//        int strideWidth = resolution.x / blockDimWidth;
//        int strideHeight = resolution.y / blockDimHeight;
//
//
//        int triangleIdxX = (blockIdx.x * blockDim.x) + threadIdx.x;
//        int triangleIdxY = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//        float weights[3];
//        //TODO Add Bounding Box
//        for (uint32_t i = (uint32_t)triangleIdxX; i < (triangleIdxX + (uint32_t)strideWidth); ++i)
//        {
//            for (uint32_t j = (uint32_t)triangleIdxX; j < (triangleIdxX +(uint32_t)strideHeight); ++j)
//            {
//                uint16_t pixelPosX = triangleIdxX + i;
//                uint16_t pixelPosY = triangleIdxY + j;
//
//
//                glm::vec2 pixelPos = glm::vec2(pixelPosX, pixelPosY);
//
//                fragment frag;
//                frag.isEmpty = false;
//                if (IsPixelInTriangle(outputTriangle, pixelPos, weights, resolution)) { //In in triangle
//
//                    float ZBufferValue = 1 / (((1 / outputTriangle.vertices[0].z) * weights[1]) + ((1 / outputTriangle.vertices[1].z) * weights[2]) + ((1 / outputTriangle.vertices[2].z) * weights[0]));
//                    float interpolatedW = 1 / (((1 / outputTriangle.vertices[0].w) * weights[1]) + ((1 / outputTriangle.vertices[1].w) * weights[2]) + ((1 / outputTriangle.vertices[2].w) * weights[0]));
//                    int pixelIdx = pixelPosX + (pixelPosY * resolution.x);
//
//
//                    bool shouldWait = true;
//                   
//                   
//                    if (ZBufferValue > 0 && ZBufferValue < 1) { //Depth test | is interpolated value inside [0,1] range
//                        fragment previousDepth = depthBuffer[pixelIdx];
//                        if (interpolatedW < previousDepth.depth) {
//                            frag.color = glm::vec3{ 255.f, 0.f, 0.f };
//                            frag.depth = interpolatedW;
//                            depthBuffer[pixelIdx] = frag;
//                        }
//                      
//
//
//                    }
//                          
//                    
//                }
//
//            }
//        }
//    
//}

void RasterizeTiles(Input_Triangle* primitives, int triangleSize, glm::vec2 resolution, fragment* depthBuffer, int* dev_mutex) {

    //RESET TILES
    int stride_x = glm::floor(resolution.x / TILE_W_AMOUNT);
    int stride_y = glm::floor(resolution.y / TILE_H_AMOUNT);
    dim3 numThreadsPerBlock(128);
    dim3 blockCount1d_tiles(((numTiles - 1) / numThreadsPerBlock.x) + 1);
    dim3 blockCount1d_triangles(((triangleSize - 1) / numThreadsPerBlock.x) + 1);
    SortTrianglesInCorrectTile << <blockCount1d_triangles, numThreadsPerBlock >> > (primitives, triangleSize, resolution, stride_x, stride_y, dev_mutex, tileBuffer);
    cudaDeviceSynchronize();
    checkCUDAError("Sorting Triangles in right Tile failed");

    //Debug in console
    //for (size_t h = 0; h < TILE_H_AMOUNT; h++)
    //{
    //    for (size_t w = 0; w < TILE_W_AMOUNT; w++)
    //    {
    //        Tile currentTile = tileBuffer[w * h];
    //        if (currentTile.m_TriangleAmount > 0)
    //            std::cout << "|";
    //        else
    //            std::cout << "-";
    //
    //    }
    //    std::cout << "\n";
    //}

   for (size_t w = 0; w < TILE_W_AMOUNT; w++)
   {
       for (size_t h = 0; h < TILE_H_AMOUNT; h++)
       {
           Tile currentTile = tileBuffer[w * h];
           if (currentTile.m_TriangleAmount > 0) {
               for (size_t x = 0; x < stride_x; x++)
               {
                   for (size_t y = 0; y < stride_y; y++)
                   {
                       int pixelidx = (x + (w * stride_x)) + (y + (h * stride_y)) * resolution.x;
                       depthBuffer[pixelidx].color = glm::vec3{ 255.f, 0.f, 0.f };
                   }
               }
           }
          
       }
   }
}



void InitializeBuffers(glm::vec2 resolution) {

    framebuffer = NULL;
    cudaError_t err = cudaMalloc((void**)&framebuffer, (int)resolution.x * (int)resolution.y * sizeof(glm::vec3));
    checkCUDAError("Init framebuffer failed");

    //set up depthbuffer
    depthbuffer = NULL;
    err = cudaMallocManaged((void**)&depthbuffer, (int)resolution.x * (int)resolution.y * sizeof(fragment));
    checkCUDAError("Init depthbuffer failed");

    int depthBufferLockSize = resolution.x * resolution.y;
    depthBufferLock = NULL;
    cudaMalloc((void**)&depthBufferLock, depthBufferLockSize * sizeof(int));
    initiateArray << <dim3(ceil((float)depthBufferLockSize / ((float)512))), dim3(512) >> > (depthBufferLock, 0, depthBufferLockSize);
    checkCUDAError("Init depthbufferLock failed");

    //Set up TileBuffert
    cudaFree(tileBuffer);
    cudaMallocManaged(&tileBuffer, numTiles * sizeof(Tile));
    cudaMemset(tileBuffer, 0, numTiles * sizeof(Tile));
    checkCUDAError("Setup TileBuffer failed");


    cudaFree(dev_tileMutex);
    cudaMalloc(&dev_tileMutex, numTiles * sizeof(int));
    checkCUDAError("Setup TileBufferMutex failed");

    MV = new glm::mat4x4;

}

__host__ void UpdateBuffers(const glm::vec3* pVertexBuffer, int vertexAnmount, const int* pIndexBuffer, int indexAmount) {

    
    if (dev_pOutputVertexBuffer != nullptr) {

        cudaFree(dev_pOutputVertexBuffer);
        cudaFree(dev_pVertexBuffer);
        cudaFree(dev_pIndexBuffer);
    }


    //Allocate vetexbuffer data in Device memory and copy from CPU memory to Device Memory
    dev_pVertexBuffer = NULL;
    cudaError err = cudaMalloc((void**)&dev_pVertexBuffer, vertexAnmount * sizeof(glm::vec3));
    checkCUDAError("Allocating vertex data on Device failed!");
    err = cudaMemcpy(dev_pVertexBuffer, pVertexBuffer, vertexAnmount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAError("Copying vertex data to Device failed!");

    dev_pOutputVertexBuffer = NULL;
    err = cudaMalloc((void**)&dev_pOutputVertexBuffer, vertexAnmount * sizeof(glm::vec4));
    checkCUDAError("Allocating output vertex data on Device failed!");

    //Allocate indexbuffer data in Device memory and copy from CPU memory to Device Memory
    dev_pIndexBuffer = NULL;
    err = cudaMalloc((void**)&dev_pIndexBuffer, indexAmount * sizeof(int));
    checkCUDAError("Allocating index data on Device failed!");
    err = cudaMemcpy(dev_pIndexBuffer, pIndexBuffer, indexAmount * sizeof(int), cudaMemcpyHostToDevice);
    checkCUDAError("Copying Index data to Device failed!");

   
}

__global__ void vertexShaderKernel(glm::vec3* pDev_vertexBuffer, glm::vec4* pDev_OutputvertexBuffer, int vertexAmount, glm::mat4x4 MV) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //Safety check
    if (index < vertexAmount) {
        //Convert to viewspace
        glm::vec4 viewSpaceP = (MV * glm::vec4(pDev_vertexBuffer[index].x, pDev_vertexBuffer[index].y, pDev_vertexBuffer[index].z, 1.f));


        pDev_OutputvertexBuffer[index] = viewSpaceP;
    }
}

__global__ void primitiveAssemblyKernel(glm::vec4* pDev_VertexBufer, int vertexAmount, int* pDev_IndexBuffer, int indexAmount, Input_Triangle* pDev_TriangleInput) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int primitivesCount = indexAmount / 3;


    if (index < primitivesCount) {

        Input_Triangle triangle;
        int f1, f2, f3;
        f1 = pDev_IndexBuffer[(index * 3)];
        f2 = pDev_IndexBuffer[(index * 3)+1];
        f3 = pDev_IndexBuffer[(index * 3)+2];



        triangle.vertices[0] = pDev_VertexBufer[f1];
        triangle.vertices[1] = pDev_VertexBufer[f2];
        triangle.vertices[2] = pDev_VertexBufer[f3];

       pDev_TriangleInput[index] = triangle;

    }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if (x <= resolution.x && y <= resolution.y) {
        image[index] = color;
    }
}
//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if (x <= resolution.x && y <= resolution.y) {
        fragment f = frag;
        buffer[index] = f;
    }
}
void ClearImage(glm::vec2 resolution)
{
    // set up thread configuration
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

    clearImage << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, framebuffer, glm::vec3(0, 0, 0));

}

void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, const glm::vec3* pVertexBuffer, int vertexAnmount, const int* pIndexBuffer, int indexAmount, glm::mat4 worldviewprojMat) {
    //Init buffers
    //set up framebuffer
    checkCUDAError("Setup failed");

    //Move Host memory To Device; Host == CPU && Device == GPU
    dev_pTriangleBuffer = NULL;

    //WorldViewProjMatrix
    glm::mat4x4* dev_MVtransform;

    //Check if we need to update buffers
    int primitiveAmount = (indexAmount / 3);
    if (primitiveAmount != m_CurrentBufferPrimitiveAmount) {

        UpdateBuffers(pVertexBuffer, vertexAnmount, pIndexBuffer, indexAmount);
    }
   

    //Allocate Triangle Buffer depending on Indices amount
    m_CurrentBufferPrimitiveAmount = primitiveAmount;
    dev_pTriangleBuffer = NULL;
    cudaError err = cudaMalloc((void**)&dev_pTriangleBuffer, m_CurrentBufferPrimitiveAmount * sizeof(Input_Triangle));
    checkCUDAError("Allocating triangle data failed!");



    // set up thread configuration
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

    //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
    clearImage << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, framebuffer, glm::vec3(0, 0, 0));
    checkCUDAError("Clearing image failed");

    //Clear depth buffer
    fragment frag;
    frag.color = glm::vec3(0, 0, 0);
    frag.normal = glm::vec3(0, 0, 0);
    frag.position = glm::vec3(0, 0, -10000);
    frag.cameraSpacePosition = glm::vec3(0, 0, -10000);
    frag.depth = 100;
    frag.isEmpty = true;
    clearDepthBuffer << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, depthbuffer, frag);

    //Vertex shader
    tileSize = 16;
    int primitiveBlocks = ceil(((float)vertexAnmount) / ((float)tileSize));
    vertexShaderKernel << <primitiveBlocks, tileSize >> > (dev_pVertexBuffer, dev_pOutputVertexBuffer, vertexAnmount, worldviewprojMat);
    //Clear TileMutex
    cudaMemset(dev_tileMutex, 0, numTiles * sizeof(int));


    //Primitive assmebly | creating the triangles
    primitiveBlocks = ceil(((float)indexAmount / 3) / ((float)tileSize));
    primitiveAssemblyKernel << <primitiveBlocks, tileSize >> > (dev_pOutputVertexBuffer, vertexAnmount, dev_pIndexBuffer, indexAmount, dev_pTriangleBuffer);
    checkCUDAError("Primitive assmebly failed");
   cudaDeviceSynchronize();


    RasterizeTiles(dev_pTriangleBuffer, primitiveAmount, resolution, depthbuffer, dev_tileMutex);
   cudaDeviceSynchronize();

    checkCUDAError("Rasterization failed");

    //Rasterization

    //Primitive assembly

   cudaDeviceSynchronize();
   sendImageToPBO << <fullBlocksPerGrid, threadsPerBlock >> > (PBOpos, resolution, depthbuffer);
   checkCUDAError("Sending To PBO failed");


    
    cudaFree(dev_pTriangleBuffer);



}

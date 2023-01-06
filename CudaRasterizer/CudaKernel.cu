
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
#include "DeviceStructs.h"

struct Tile {
    int m_TriangleIndexes[10000]; //Indexes of triangles that are in this Tile
    int m_TriangleAmount{ 0 };
};

__device__ int* depthBufferLock;

__device__  int* dev_pIndexBuffer = nullptr;
__device__  glm::vec3* dev_pVertexBuffer = nullptr;
__device__  glm::vec4* dev_pOutputVertexBuffer = nullptr;
__device__  Input_Triangle* dev_pTriangleBuffer = nullptr;
__device__ glm::vec3* framebuffer;

//Lighting information
__device__ __constant__ const float3 Dev_lightDirection{ 0.577f , -0.577f, -0.577f };
__device__ __constant__ const float3 Dev_lightColor{ 1.f,1.f,1.f };
__device__ __constant__ const float Dev_LightIntensity = 3.f;
__device__ __constant__ const float Dev_PI = 3.141592654f;


Tile* tileBuffer = NULL;
int* dev_tileMutex = NULL;
fragment* depthbuffer;
glm::mat4x4* MV;

#define TILE_H_AMOUNT 5
#define TILE_W_AMOUNT 5
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
    return 0.5 * ((tri.viewspaceCoords[2].x - tri.viewspaceCoords[0].x) * (tri.viewspaceCoords[1].y - tri.viewspaceCoords[0].y)
        - (tri.viewspaceCoords[1].x - tri.viewspaceCoords[0].x) * (tri.viewspaceCoords[2].y - tri.viewspaceCoords[0].y));
}
__device__ float Cross(glm::vec2 lhs, glm::vec2 rhs) {
    return lhs.x * rhs.y - lhs.y * rhs.x;
}
__device__ bool IsPixelInTriangle(Input_Triangle tri, const glm::vec2 point, float* pWeight, glm::vec2 resolution) {
    //Maybe positive values get passed this too

    glm::vec2 ScreenspaceA = glm::vec2{ tri.Screenspace[0] };
    glm::vec2 ScreenspaceB = glm::vec2{ tri.Screenspace[1] };
    glm::vec2 ScreenspaceC = glm::vec2{ tri.Screenspace[2] };

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
__host__ __device__ void getAABBForTriangle(Input_Triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint) {
    minpoint = glm::vec3(min(min(tri.Screenspace[0].x, tri.Screenspace[1].x), tri.Screenspace[2].x),
        min(min(tri.Screenspace[0].y, tri.Screenspace[1].y), tri.Screenspace[2].y),
        min(min(tri.Screenspace[0].z, tri.Screenspace[1].z), tri.Screenspace[2].z));
    maxpoint = glm::vec3(max(max(tri.Screenspace[0].x, tri.Screenspace[1].x), tri.Screenspace[2].x),
        max(max(tri.Screenspace[0].y, tri.Screenspace[1].y), tri.Screenspace[2].y),
        max(max(tri.Screenspace[0].z, tri.Screenspace[1].z), tri.Screenspace[2].z));
}
__global__ void SortTrianglesInCorrectTile(Input_Triangle* primitives, int triangleSize, glm::vec2 resolution, int strideX, int strideY, int* dev_tilemutex, Tile* dev_tilebuffer) {
    int triangleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (triangleIndex < triangleSize) {
        Input_Triangle input = primitives[triangleIndex];
            AABB boundingBox = getBBForTriangle(input);


            int tileminX = glm::floor(boundingBox.min.x / strideX);
            int tilemaxX = glm::ceil(boundingBox.max.x / strideX);
            int tileminY = glm::floor(boundingBox.min.y / strideY);
            int tilemaxY = glm::ceil(boundingBox.max.y / strideY);

            int tilesX = (int)glm::ceil(double(resolution.x) / double(strideX));
            int tilesy = (int)glm::ceil(double(resolution.y) / double(strideY));

            //cliping
            if (tilemaxY > tilesy || tileminY < 0 || tilemaxX > tilesX || tileminX < 0)
                return;

            

            for (int i = tileminY; i < tilemaxY; i++)
            {
                for (int j = tileminX; j < tilemaxX; j++)
                {
                    int tileID = j + i * tilesX;
                    bool isSet = false;
                    do
                    {
                        int oldValue = atomicCAS(&dev_tilemutex[tileID], 0, 1);
                        isSet = oldValue == 0;
                        if (isSet)
                        {

                            //Critical sectie
                            int t = dev_tilebuffer[tileID].m_TriangleAmount;
                            dev_tilebuffer[tileID].m_TriangleIndexes[t] = triangleIndex;
                            dev_tilebuffer[tileID].m_TriangleAmount = t + 1;
                            dev_tilemutex[tileID] = 0;
                        }

                    } while (!isSet);
                }
            }

       
    }
}
__global__
void resetTiles(int numTiles, int stride_x, int stride_y, Tile* dev_tiles)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numTiles)
    {
        for (int i = 0; i < dev_tiles[index].m_TriangleAmount; i++)
        {
            dev_tiles[index].m_TriangleIndexes[i] = 0;
        }
        dev_tiles[index].m_TriangleAmount = 0;
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
__device__
float Cuda_dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
glm::vec3 PixelShading(glm::vec2 pixePos, Input_Triangle* CurrTriangle) {
    glm::vec3 lightDir;            
    lightDir.x = Dev_lightDirection.x;
    lightDir.y = Dev_lightDirection.y;
    lightDir.z = Dev_lightDirection.z;

    CurrTriangle->Normal.y = -CurrTriangle->Normal.y;
    //Shade side of cube
    float intensity;
    if (CurrTriangle->Normal.y > 0)
        intensity = 1.f;
    else if (-CurrTriangle->Normal.y < 0)
        intensity = 0.4f;
    else if (CurrTriangle->Normal.x != 0)
        intensity = 0.8f;
    else if (CurrTriangle->Normal.z != 0)
        intensity = 0.6f;

    float lambertCosine = glm::dot(lightDir, CurrTriangle->Normal);
    glm::vec3 LambertColor = (CurrTriangle->Color * Dev_LightIntensity) / Dev_PI; //BRDF LAMBERT
    return LambertColor * intensity * glm::clamp(lambertCosine, 0.2f, 1.f);

}
__global__
void RasterizePixels(int pixelXoffset, int pixelYoffset, int numpixelsX, int numpixelsY,
    glm::vec2 resolution, int tileID, Tile* dev_tilesList,
    Input_Triangle* dev_primitives, fragment* dev_depthBuffer)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = (x + pixelXoffset) + ((y + pixelYoffset) * resolution.x);


    if (x < numpixelsX && y < numpixelsY)
    {
        //Each thread loops over the triangles inside the tile
        //Discard tiles (ie kernel launches) that dont have any triangles inside them --> implicitly done by for loop
        for (int i = 0; i < dev_tilesList[tileID].m_TriangleAmount; i++)
        {
          
            int triangleIndex = dev_tilesList[tileID].m_TriangleIndexes[i];
            Input_Triangle currentTriangle = dev_primitives[triangleIndex];

            int _x = (x + pixelXoffset);
            int _y = (y + pixelYoffset);

            float weights[3];
            if (IsPixelInTriangle(currentTriangle, glm::vec2{_x, _y}, weights, resolution)) { //In in triangle
               
                float totalWeight = weights[0] + weights[1] + weights[2];

                float ZBufferValue = 1 / (((1 / currentTriangle.NDC[0].z) * weights[1]) + ((1 / currentTriangle.NDC[1].z) * weights[2])
                    + ((1 / currentTriangle.NDC[2].z) * weights[0]));

                float interpolatedW = 1 / (((1 / currentTriangle.NDC[0].w) * weights[1]) + ((1 / currentTriangle.NDC[1].w) * weights[2])
                    + ((1 / currentTriangle.NDC[2].w) * weights[0]));


                if (ZBufferValue > 0 && ZBufferValue < 1) { //Depth test | is interpolated value inside [0,1] range
                    fragment previousDepth = dev_depthBuffer[index];
                    if (interpolatedW < previousDepth.depth) {
                        fragment frag;
                        frag.color = PixelShading(glm::vec2{ _x, _y }, &currentTriangle);
                        frag.depth = interpolatedW;
                        dev_depthBuffer[index] = frag;

                    }




                }
            }
        }
    }
}
void RasterizeTiles(Input_Triangle* primitives, int triangleSize, glm::vec2 resolution, fragment* depthBuffer, int* dev_mutex) {

    cudaDeviceSynchronize();

    //RESET TILES
    int stride_x = glm::floor(resolution.x / TILE_W_AMOUNT);
    int stride_y = glm::floor(resolution.y / TILE_H_AMOUNT);
    dim3 numThreadsPerBlock(128);
    
    //Clear TileMutex
    cudaMemset(dev_tileMutex, 0, numTiles * sizeof(int));
    dim3 blockCount1d_tiles(((numTiles - 1) / numThreadsPerBlock.x) + 1);
    resetTiles << <blockCount1d_tiles, numThreadsPerBlock >> > (numTiles, stride_x, stride_y, tileBuffer);

    cudaDeviceSynchronize();

    dim3 blockCount1d_triangles(((triangleSize - 1) / numThreadsPerBlock.x) + 1);
    SortTrianglesInCorrectTile << <blockCount1d_triangles, numThreadsPerBlock >> > (primitives, triangleSize, resolution, stride_x, stride_y, dev_mutex, tileBuffer);
    cudaDeviceSynchronize();
    checkCUDAError("Sort triangles failed");


    int sideLength2d = 25;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    
    int tilesX = glm::ceil(resolution.x / stride_x);
    int tileXcount = 0;
    int tileYcount = 0;
    for (int i = 0; i < resolution.x; i += stride_y)
    {
        for (int j = 0; j < resolution.y; j += stride_x)
        {
            //Calculate tile pixel positions
            int tileID = tileXcount + tileYcount * (tilesX);
            glm::ivec2 pixelMin = glm::ivec2(tileXcount * stride_x, tileYcount * stride_y);
            glm::ivec2 pixelMax = glm::ivec2(glm::min((tileXcount + 1) * stride_x, int(resolution.x - 1)), glm::min((tileYcount + 1) * stride_y, int(resolution.y - 1)));
        
            
            //Calculate tile pixel distances
            int numpixelsX = pixelMax.x - pixelMin.x;
            int numpixelsY = pixelMax.y - pixelMin.y;
    
            int pixelXoffset = tileXcount * stride_x;
            int pixelYoffset = tileYcount * stride_y;
    
            dim3 blockCount2d_tilePixels((numpixelsX - 1) / blockSize2d.x + 1,
                (numpixelsY - 1) / blockSize2d.y + 1);
    
            RasterizePixels << <blockCount2d_tilePixels, blockSize2d >> > (pixelXoffset, pixelYoffset,
                numpixelsX, numpixelsY, resolution,tileID , tileBuffer, primitives, depthBuffer);
    
            checkCUDAError("tile rasterization failed");
    
            tileXcount++;
        }
        tileXcount = 0;
        tileYcount++;
    }
    cudaDeviceSynchronize();
    checkCUDAError("full rasterization failed");

    ////Debug in console
    //for (size_t h = 0; h < TILE_H_AMOUNT; h++)
    //{
    //    for (size_t w = 0; w < TILE_W_AMOUNT; w++)
    //    {
    //        Tile currentTile = tileBuffer[w + (h * TILE_W_AMOUNT)];
    //        if (currentTile.m_TriangleAmount > 0)
    //            std::cout << " " << currentTile.m_TriangleAmount << " ";
    //        else
    //            std::cout << " 00 ";
    //
    //    }
    //    std::cout << "\n";
    //}
   // std::cout << "XDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD\n";
   //for (size_t w = 0; w < TILE_W_AMOUNT; w++)
   //{
   //    for (size_t h = 0; h < TILE_H_AMOUNT; h++)
   //    {
   //        Tile currentTile = tileBuffer[w + (h* TILE_W_AMOUNT)];
   //        if (currentTile.m_TriangleAmount > 0) {
   //            for (size_t x = 0; x < stride_x; x++)
   //            {
   //                for (size_t y = 0; y < stride_y; y++)
   //                {
   //                    int pixelidx = (x + w * stride_x) + ((y + h * stride_y) * resolution.x);
   //                    depthBuffer[pixelidx].color = glm::vec3{ 255.f, 0.f, 0.f };
   //                }
   //            }
   //        }
   //       
   //    }
   //}
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
}

__host__ void UpdateBuffers(const glm::vec3* pVertexBuffer, int vertexAnmount, const int* pIndexBuffer, int indexAmount) {
    //Random device



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
        glm::vec4 viewSpaceP = (MV * glm::vec4(pDev_vertexBuffer[index].x, pDev_vertexBuffer[index].y, pDev_vertexBuffer[index].z, 1.f) );


        pDev_OutputvertexBuffer[index] = viewSpaceP;
    }
}

__global__ void primitiveAssemblyKernel(glm::vec4* pDev_ViewSpaceVertexBufer, 
    glm::vec3* pDev_WorldSpaceVertexBufer, int vertexAmount, int* pDev_IndexBuffer, int indexAmount, Input_Triangle* pDev_TriangleInput, glm::vec2 resolution)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int primitivesCount = indexAmount / 3;


    if (index < primitivesCount) {

        Input_Triangle triangle;
        int f1, f2, f3;


        f1 = pDev_IndexBuffer[(index * 3) + 2];
        f2 = pDev_IndexBuffer[(index * 3)+1];
        f3 = pDev_IndexBuffer[(index * 3)];

        triangle.viewspaceCoords[0] = pDev_ViewSpaceVertexBufer[f1];
        triangle.viewspaceCoords[1] = pDev_ViewSpaceVertexBufer[f2];
        triangle.viewspaceCoords[2] = pDev_ViewSpaceVertexBufer[f3];

        triangle.worldSpaceCoords[0] = pDev_WorldSpaceVertexBufer[f1];
        triangle.worldSpaceCoords[1] = pDev_WorldSpaceVertexBufer[f2];
        triangle.worldSpaceCoords[2] = pDev_WorldSpaceVertexBufer[f3];

        //Calculate triangle Normal
        glm::vec3 edgeA = (triangle.worldSpaceCoords[1] - triangle.worldSpaceCoords[0]);
        glm::vec3 edgeB = (triangle.worldSpaceCoords[2] - triangle.worldSpaceCoords[1]);
        glm::vec3 normal = -glm::normalize(glm::cross(edgeA, edgeB));

        triangle.Normal = normal;

        triangle.Normal = normal;
        int xRange = (int)pDev_WorldSpaceVertexBufer[f1].x % 5;
        int yRange = (int)pDev_WorldSpaceVertexBufer[f1].y % 5;
        int zRange = (int)pDev_WorldSpaceVertexBufer[f1].z % 5;

        triangle.Color = glm::vec3{ xRange * 70, yRange * 70, zRange * 70 };

        //if (index % 5 == 0) {
        //    triangle.Color = glm::vec3{ 255.f, 0.f, 0.f };
        //}
        //else  if (index % 5 == 1) {
        //    triangle.Color = glm::vec3{ 0.f, 255.f, 0.f };
        //}
        //else  if (index % 5 == 2) {
        //    triangle.Color = glm::vec3{ 0.f, 0.f, 255.f };
        //}
        //else  if (index % 5 == 3) {
        //    triangle.Color = glm::vec3{ 0.f, 255.f, 255.f };
        //}
        //else  if (index % 5 == 5) {
        //    triangle.Color = glm::vec3{ 255.f, 0.f, 255.f };
        //}
        //printf("%f, %f, %f", triangle.Normal.x, triangle.Normal.y, triangle.Normal.z);
        //printf("\n");

        for (size_t i = 0; i < 3; i++)
        {

            //Convert to NDC coordinates
            glm::vec4 NDC = glm::vec4(triangle.viewspaceCoords[i].x / triangle.viewspaceCoords[i].w
                , triangle.viewspaceCoords[i].y / triangle.viewspaceCoords[i].w, 
                  triangle.viewspaceCoords[i].z / triangle.viewspaceCoords[i].w
                , triangle.viewspaceCoords[i].w);
           
            triangle.NDC[i] = NDC;

            //Convert To Screenspace
            glm::vec3 screenSpace;

            screenSpace.x = ((NDC.x + 1.f)) * resolution.x * 0.5f;
            screenSpace.y = ((1.f - NDC.y)) * resolution.y * 0.5f;
            screenSpace.z = ((NDC.z + 1.0f) * 0.5f);
            triangle.Screenspace[i] = screenSpace;




        }

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



    //Primitive assmebly | creating the triangles
    primitiveBlocks = ceil(((float)indexAmount / 3) / ((float)tileSize));
    primitiveAssemblyKernel << <primitiveBlocks, tileSize >> > (dev_pOutputVertexBuffer, dev_pVertexBuffer,  vertexAnmount, dev_pIndexBuffer, indexAmount, dev_pTriangleBuffer, resolution);
    checkCUDAError("Primitive assmebly failed");


    RasterizeTiles(dev_pTriangleBuffer, primitiveAmount, resolution, depthbuffer, dev_tileMutex);

    checkCUDAError("Rasterization failed");

    //Rasterization

    //Primitive assembly

   sendImageToPBO << <fullBlocksPerGrid, threadsPerBlock >> > (PBOpos, resolution, depthbuffer);
   checkCUDAError("Sending To PBO failed");


   cudaDeviceSynchronize();
    cudaFree(dev_pTriangleBuffer);



}

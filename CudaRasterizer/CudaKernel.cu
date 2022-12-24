
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

__device__  int* dev_pIndexBuffer;
__device__  glm::vec3* dev_pVertexBuffer;
__device__  glm::vec4* dev_pOutputVertexBuffer;

__device__  Input_Triangle* dev_pTriangleBuffer;

__device__ glm::vec3* framebuffer;
fragment* depthbuffer;
glm::mat4x4* MV;

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

//Find bounding box
 __device__ void getBoundingBoxForTriangle(Output_Triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint) {
    minpoint = glm::vec3(min(min(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x),
        min(min(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y),
        min(min(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z));

    maxpoint = glm::vec3(max(max(tri.vertices[0].x, tri.vertices[1].x), tri.vertices[2].x),
        max(max(tri.vertices[0].y, tri.vertices[1].y), tri.vertices[2].y),
        max(max(tri.vertices[0].z, tri.vertices[1].z), tri.vertices[2].z));
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
void RasterizeKernel(Input_Triangle* primitives, int triangleSize, glm::vec2 resolution, fragment* depthBuffer, glm::mat4x4* worldviewprojMat) {
    int triangleIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (triangleIdx < triangleSize) {
        Input_Triangle currTriangle = primitives[triangleIdx];
        Output_Triangle outputTriangle;

        //Transform vertices
        for (size_t i = 0; i < 3; i++)
        {
            
            //Convert to NDC coordinates
            glm::vec4 NDC = glm::vec4(currTriangle.vertices[i].x / currTriangle.vertices[i].w, currTriangle.vertices[i].y / currTriangle.vertices[i].w, currTriangle.vertices[i].z / currTriangle.vertices[i].w, currTriangle.vertices[i].w);
            
            //if (NDC.x > 1 || NDC.x < -1 || NDC.y > 1 || NDC.y < -1)
                //return;

             outputTriangle.vertices[i] = NDC;
        }


        glm::vec3 Min, Max;
        getBoundingBoxForTriangle(outputTriangle, Min, Max);
        float pixelWidth = 1.0f / (float)resolution.x;
        float pixelHeight = 1.0f / (float)resolution.y;
        float weights[3]{};



        //TODO Add Bounding Box
        for (int i =0; i < (resolution.x) ; i += 1)
        {
            for (int j = 0; j < (resolution.y); j += 1)
            {
              
               glm::vec2 pixelPos = glm::vec2(i, j);

               fragment frag;
               frag.isEmpty = false;
               if (IsPixelInTriangle(outputTriangle, pixelPos,weights, resolution)) { //In in triangle

                   float ZBufferValue = 1 / (((1 / outputTriangle.vertices[0].z) * weights[1]) + ((1 / outputTriangle.vertices[1].z) * weights[2]) + ((1 / outputTriangle.vertices[2].z) * weights[0]));
                   float interpolatedW = 1 / (((1 / outputTriangle.vertices[0].w) * weights[1]) + ((1 / outputTriangle.vertices[1].w) * weights[2]) + ((1 / outputTriangle.vertices[2].w) * weights[0]));
                   int pixelIdx = i + (j * resolution.x);


                   if (ZBufferValue > 0 && ZBufferValue < 1) { //Depth test | is interpolated value inside [0,1] range
                       fragment previousDepth = depthBuffer[pixelIdx];
                       if (interpolatedW < previousDepth.depth) {
                           bool shouldWait = true;
                           frag.color = glm::vec3{ 255.f, 0.f, 0.f };
                           frag.depth = interpolatedW;
                           depthBuffer[pixelIdx] = frag;
                       }

                  
                       

                   }
               }

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

    MV = new glm::mat4x4;

}
__global__ void vertexShaderKernel(glm::vec3* pDev_vertexBuffer, glm::vec4* pDev_OutputvertexBuffer, int vertexAmount, glm::mat4x4* MV) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //Safety check
    if (index < vertexAmount) {
        //Convert to viewspace
        glm::vec4 viewSpaceP = (*MV * glm::vec4(pDev_vertexBuffer[index].x, pDev_vertexBuffer[index].y, pDev_vertexBuffer[index].z, 1.f));


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
    *MV = worldviewprojMat;
    cudaError_t err = cudaMalloc((void**)&dev_MVtransform, sizeof(glm::mat4x4));
    err = cudaMemcpy(dev_MVtransform, MV, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);

    //Allocate Triangle Buffer depending on Indices amount
    dev_pTriangleBuffer = NULL;
    int primitiveAmount = (indexAmount / 3);
    err = cudaMalloc((void**)&dev_pTriangleBuffer, primitiveAmount * sizeof(Input_Triangle));
    checkCUDAError("Allocating triangle data failed!");
    //Allocate vetexbuffer data in Device memory and copy from CPU memory to Device Memory
    dev_pVertexBuffer = NULL;
    err = cudaMalloc((void**)&dev_pVertexBuffer, vertexAnmount * sizeof(glm::vec3));
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



    // set up thread configuration
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

    //Clear depth buffer
    fragment frag;
    frag.color = glm::vec3(0, 0, 0);
    frag.normal = glm::vec3(0, 0, 0);
    frag.position = glm::vec3(0, 0, -10000);
    frag.cameraSpacePosition = glm::vec3(0, 0, -10000);
    frag.depth = 100;
    frag.isEmpty = true;
    clearDepthBuffer << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, depthbuffer, frag);
    cudaDeviceSynchronize();

    //Vertex shader
    tileSize = 16;
    int primitiveBlocks = ceil(((float)vertexAnmount) / ((float)tileSize));
    vertexShaderKernel << <primitiveBlocks, tileSize >> > (dev_pVertexBuffer, dev_pOutputVertexBuffer, vertexAnmount, dev_MVtransform);
    cudaDeviceSynchronize();


    //Primitive assmebly | creating the triangles
    primitiveBlocks = ceil(((float)indexAmount / 3) / ((float)tileSize));
    primitiveAssemblyKernel << <primitiveBlocks, tileSize >> > (dev_pOutputVertexBuffer, vertexAnmount, dev_pIndexBuffer, indexAmount, dev_pTriangleBuffer);
    cudaDeviceSynchronize();


    RasterizeKernel <<<fullBlocksPerGrid, threadsPerBlock>>>(dev_pTriangleBuffer, primitiveAmount, resolution, depthbuffer, dev_MVtransform);
    cudaDeviceSynchronize();

    checkCUDAError("Rasterization failed");

    //Rasterization

    //Primitive assembly


   render << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, depthbuffer, framebuffer, depthbuffer);
   checkCUDAError("Render failed");
   sendImageToPBO << <fullBlocksPerGrid, threadsPerBlock >> > (PBOpos, resolution, framebuffer);
   checkCUDAError("Sending To PBO failed");


    cudaDeviceSynchronize();
    cudaFree(dev_pOutputVertexBuffer);
    cudaFree(dev_pTriangleBuffer);
    cudaFree(dev_pVertexBuffer);
    cudaFree(dev_pIndexBuffer);





}

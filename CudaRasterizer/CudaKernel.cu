
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
__device__ Input_Triangle* trDeviceArray;
__device__ glm::vec3* framebuffer;
fragment* depthbuffer;
cudaMat4* MV;

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
cudaMat4 glmMat4ToCudaMat4(glm::mat4 a) {
    //swap from glm column major to cuda row major
    cudaMat4 m; a = glm::transpose(a);
    m.x = a[0];
    m.y = a[1];
    m.z = a[2];
    m.w = a[3];
    return m;
}
__host__ __device__ glm::vec4 multiplyMV4(cudaMat4 m, glm::vec4 v) {
    glm::vec4 r(1, 1, 1, 1);
    r.x = (m.x.x * v.x) + (m.x.y * v.y) + (m.x.z * v.z) + (m.x.w * v.w);
    r.y = (m.y.x * v.x) + (m.y.y * v.y) + (m.y.z * v.z) + (m.y.w * v.w);
    r.z = (m.z.x * v.x) + (m.z.y * v.y) + (m.z.z * v.z) + (m.z.w * v.w);
    r.w = (m.w.x * v.x) + (m.w.y * v.y) + (m.w.z * v.z) + (m.w.w * v.w);
    return r;
}

__global__
void RasterizeKernel(Input_Triangle* primitives, int triangleSize, glm::vec2 resolution, fragment* depthBuffer, cudaMat4* worldviewprojMat) {
    int triangleIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (triangleIdx < triangleSize) {
        Input_Triangle currTriangle = primitives[triangleIdx];
        Output_Triangle outputTriangle;

        //To viewspace
        glm::vec4 viewSpaceP0 = multiplyMV4(*worldviewprojMat , glm::vec4(currTriangle.vertices[0], 1.f));
        glm::vec4 viewSpaceP1=  multiplyMV4(*worldviewprojMat , glm::vec4(currTriangle.vertices[1], 1.f));
        glm::vec4 viewSpaceP3 = multiplyMV4(*worldviewprojMat , glm::vec4(currTriangle.vertices[2], 1.f));

        //To NDC 
        outputTriangle.vertices[0] = glm::vec4(viewSpaceP0.x / viewSpaceP0.z, viewSpaceP0.y / viewSpaceP0.z, -viewSpaceP0.z, viewSpaceP0.w);
        outputTriangle.vertices[1] = glm::vec4(viewSpaceP1.x / viewSpaceP1.z, viewSpaceP1.y / viewSpaceP1.z, -viewSpaceP1.z, viewSpaceP1.w);
        outputTriangle.vertices[2] = glm::vec4(viewSpaceP3.x / viewSpaceP3.z, viewSpaceP3.y / viewSpaceP3.z, -viewSpaceP3.z, viewSpaceP3.w);


        glm::vec3 Min, Max;
        getBoundingBoxForTriangle(outputTriangle, Min, Max);
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
               if (IsPixelInTriangle(outputTriangle, pixelPos,weights, resolution)) { //In in triangle

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

    MV = new cudaMat4;

}

void kernelCleanup()
{
    cudaFree(trDeviceArray);
    cudaFree(framebuffer);
    cudaFree(depthbuffer);
    cudaFree(depthBufferLock);
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

void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution,const Input_Triangle* TrArray, int TriangleSize, glm::mat4 worldviewprojMat) {
    //Init buffers
    //set up framebuffer
   
    checkCUDAError("Setup failed");

    //Move Host memory To Device; Host == CPU && Device == GPU
   

    trDeviceArray = NULL;


    cudaMat4* dev_MVtransform;
    *MV = glmMat4ToCudaMat4(worldviewprojMat);
    cudaMalloc((void**)&dev_MVtransform, sizeof(cudaMat4));
    cudaMemcpy(dev_MVtransform, MV, sizeof(cudaMat4), cudaMemcpyHostToDevice);
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaError_t err = cudaMallocManaged((void**)&trDeviceArray, TriangleSize * sizeof(Input_Triangle));
    err = cudaMemcpy(trDeviceArray, TrArray, TriangleSize * sizeof(Input_Triangle), cudaMemcpyHostToDevice);
    checkCUDAError("Copying Triangle data failed");

    // set up thread configuration
    int tileSize = 8;
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 fullBlocksPerGrid((int)ceil(float(resolution.x) / float(tileSize)), (int)ceil(float(resolution.y) / float(tileSize)));

    fragment frag;
    frag.color = glm::vec3(0, 0, 0);
    frag.normal = glm::vec3(0, 0, 0);
    frag.position = glm::vec3(0, 0, -10000);
    frag.cameraSpacePosition = glm::vec3(0, 0, -10000);
    frag.isEmpty = true;
    clearImage << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, framebuffer, glm::vec3(0, 0, 0));
    clearDepthBuffer << <fullBlocksPerGrid, threadsPerBlock >> > (resolution, depthbuffer, frag);
    cudaDeviceSynchronize();
    RasterizeKernel <<<fullBlocksPerGrid, threadsPerBlock>>>(trDeviceArray, TriangleSize, resolution, depthbuffer, dev_MVtransform);
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

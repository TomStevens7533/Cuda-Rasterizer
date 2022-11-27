
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>

using namespace std::chrono;

void add(int n, float* x, float* y) 
{

	for (size_t i = 0; i < n; i++)
	{
		y[i] = x[i] + y[i];
	}
}

__global__
void parallelAdd(int n, float* x, float* y)
{
	int index = threadIdx.x;
	int stride = blockDim.x;

	for (size_t i = index; i < n; i+= stride)
	{
		y[i] = x[i] + y[i];

	}
}

int dsdamain() {
	int N = 1 << 20;

	//Alocate cpu only memory
	float* x = new float[N];
	float* y = new float[N];

	for (size_t i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	//Allocate shared unified memory with GPU

	float* Parx;
	float* Pary;
	cudaMallocManaged(&Parx, N * sizeof(float));
	cudaMallocManaged(&Pary, N * sizeof(float));

	for (size_t i = 0; i < N; i++)
	{
		Parx[i] = 1.0f;
		Pary[i] = 2.0f;
	}
	

	auto start = high_resolution_clock::now();
	add(N, x, y);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by non-parralel function: "
		<< duration.count() << " microseconds" << std::endl;


	start = high_resolution_clock::now();
	

	//<<<N,X>>> N = amount of threadblocks; X = amount of threads in a block;
	parallelAdd<<<1,N>>>(N, x, y);
	cudaDeviceSynchronize();
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << "Time taken by parralel function: "
		<< duration.count() << " microseconds" << std::endl;

	delete[] x;
	delete[] y;

	//free cuda memory wait till kernel is done
	cudaFree(Parx);
	cudaFree(Pary);

	return 0;

}
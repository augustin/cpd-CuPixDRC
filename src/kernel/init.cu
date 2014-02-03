#include "init.h"
#include "kernel.h"

#include "errors.h"

#ifdef CUDA
int* kernel_main_cuda(int device)
{
    int* ret = (int*)malloc(sizeof(int)*3*MAX_ERRORS);
    memset((char*)ret, '\0', sizeof(int)*3*MAX_ERRORS);

    int w = 10, h = 10;
    char* pixels =
        "xxxxxxxxxx"
        "xxxxxxxxxx"
        "xxxxxxxxxx"
        "xxxxxxxxxx"
        "xxxxx    x"
        "xxxxxxxxxx"
        "xxx xxxxxx"
        "xxx xxxxxx"
        "xxx xxxxxx"
        "xxx xxxxxx";

    cudaSetDevice(device);

    char* devPixels;
    int* error_buffer;
    cudaMalloc((void**)&devPixels, w*h);
    cudaMalloc((void**)&error_buffer, sizeof(int)*3*MAX_ERRORS);

    cudaMemset(error_buffer, 0, sizeof(int)*3*MAX_ERRORS);

    cudaMemcpy(devPixels, pixels, w*h, cudaMemcpyHostToDevice);

    dim3 dimBlock(blocksize, 1);
    dim3 dimGrid(1, 1);

    device_drc<<<1, 1>>>(devPixels, w, h, error_buffer);

    cudaMemcpy(ret, error_buffer, sizeof(int)*3*MAX_ERRORS, cudaMemcpyDeviceToHost);
    cudaFree(error_buffer);
    cudaFree(devPixels);

    return ret;
}
#else
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

int* kernel_main_cpu()
{
	int* ret = (int*)malloc(sizeof(int)*3*MAX_ERRORS);
	memset((char*)ret, '\0', sizeof(int)*3*MAX_ERRORS);
	char* testarray =
		"xxxxxxxxxx"
		"xxxxxxxxxx"
		"xxxxxxxxxx"
		"xxxxxxxxxx"
		"xxxxx    x"
		"xxxxxxxxxx"
		"xxx xxxxxx"
		"xxx xxxxxx"
		"xxx xxxxxx"
		"xxx xxxxxx";

	cpu_drc(testarray, 10, 10, ret);
	return ret;
}
#endif

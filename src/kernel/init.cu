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

    /* NOTES:
     * 1. Warps (blocks) come in multiples of 32, so make sure your block size
     *    is a multiple of 32, or else the remaining SMP cores will be wasted.
     * 2. Threads determine what check type they're doing based on what their
     *    thread ID is (bX*bY*tX*tY), the image's dimensions, and total thread
     *    count. Yes, the algorithim works on non-square images, and it uses
     *    SYNCTHREADS so that the warp proceeds to the DRC all at once.
     * 3. The maximum thread count is 2048, but I highly doubt it's possible to
     *    write something that can use that many threads without deadlocking.
     *    Plus this usecase has no need for that.
     * 4. An acceptable way to determine optimum thread count (remember, thread-
     *    count is blocks*threads) is to [TODO]
     * 5. Each thread is limited to MAX_ERRORS/<numthreads> errors, so there's
     *    no memory overflow. If you need more errors than that, your file is
     *    too buggy and it's not my fault.
     */

    dim3 blocks(8, 8);
    dim3 threads(1, 1);

    device_drc<<<blocks, threads>>>(devPixels, w, h, error_buffer);

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

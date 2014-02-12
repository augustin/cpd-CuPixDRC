#include "init.h"
#include "kernel.h"

#include "errors.h"

#ifdef CUDA
int* kernel_main_cuda(int device, const char* pixels, int w, int h)
{
    int* ret = (int*)malloc(sizeof(int)*3*MAX_ERRORS);
    memset((char*)ret, '\0', sizeof(int)*3*MAX_ERRORS);

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
     * 2. Each thread is limited to MAX_ERRORS/<numthreads> errors, so there's
     *    no memory overflow. If you need more errors than that, your file is
     *    too buggy and it's not my fault.
     * 3. Each thread schedules it's own checks using just it's thread ID and
     *    the dimensions of the image. All threads do all horizontal checks, and
     *    then all move on to vertical checks at the same time.
     * 4. Each thread does the row in the image that corresponds with its thread
     *    ID, then increments rows by the number of threads until it reaches the
     *    end. The same is done for vertical checks.
     * 5. The maximum thread count is 65,535, but I highly doubt it's possible to
     *    write something that can use that many threads without deadlocking.
     *    Plus this usecase has no need for that.
     */

    //dim3 blocks(32);
    //dim3 threads(6, 6);

    device_drc<<<64, 1>>>(devPixels, w, h, error_buffer);

    cudaMemcpy(ret, error_buffer, sizeof(int)*3*MAX_ERRORS, cudaMemcpyDeviceToHost);
    cudaFree(error_buffer);
    cudaFree(devPixels);

    return ret;
}
#else
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

int* kernel_main_cpu(const char* pixels, int w, int h)
{
    int* ret = (int*)malloc(sizeof(int)*3*MAX_ERRORS);
    memset((char*)ret, '\0', sizeof(int)*3*MAX_ERRORS);

    cpu_drc(pixels, w, h, ret);
    return ret;
}
#endif

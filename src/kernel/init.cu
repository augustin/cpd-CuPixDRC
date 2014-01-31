// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "init.h"
#include "kernel.h"

#ifdef CUDA
int* kernel_main_cuda(int device)
{
    char a[N] = "Hello \0\0\0\0\0\0";
    int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    char *ad;
    int *bd;
    const int csize = N*sizeof(char);
    const int isize = N*sizeof(int);

    printf("%s", a);

    cudaMalloc( (void**)&ad, csize );
    cudaMalloc( (void**)&bd, isize );
    cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
    cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

    cudaSetDevice(device);
    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( 1, 1 );
	//device_drc<<<dimGrid, dimBlock>>>(*ad, 1, 1);
    cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
    cudaFree( ad );
    cudaFree( bd );

    printf("%s\n", a);
}
#else
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

#ifndef KERNEL_DEFS_H
#define KERNEL_DEFS_H

#if defined(CUDA)
#   define SMP
#   define SYNCTHREADS __syncthreads()

#	define KERNEL_FUNCTION(type, name) __global__ type device_##name

    /* This design is idiotic, stupid, and confusing.
     *  - blockDim is the number of THREADS in a block
     *  - gridDim  is the number of BLOCKS in the grid
     * BUT:
     *  - threadIdx is the ID of the THREAD in the BLOCK
     *  - blockIdx  is the ID of the BLOCK in the GRID
     */
#   define THREADS_X (gridDim.x*blockDim.x)
#   define THREADS_Y (gridDim.y*blockDim.y)

#   define THREAD_ID_X (blockIdx.x*blockDim.x+threadIdx.x)
#   define THREAD_ID_Y (blockIdx.y*blockDim.y+threadIdx.y)
#else
#	define KERNEL_FUNCTION(type, name) type cpu_##name

#   define THREADS_X 1
#   define THREADS_Y 1

#   define THREAD_ID_X 0
#   define THREAD_ID_Y 0
#endif

#define THREAD_ID (THREADS_X*THREAD_ID_Y + THREAD_ID_X)
#define THREADS_TOT (THREADS_X*THREADS_Y)

#endif // KERNEL_DEFS_H

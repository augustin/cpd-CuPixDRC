#ifndef KERNEL_KERNEL_H
#define KERNEL_KERNEL_H

/* Arch-specific macros. */
#if defined(CUDA)
    #define SMP
    #define SYNCTHREADS __syncthreads()

    #define KERNEL_FUNCTION(type, name) __global__ type device_##name

    /* This API design is *very* confusing.
     *  - blockDim is the number of THREADS in a block
     *  - gridDim  is the number of BLOCKS in the grid
     * BUT:
     *  - threadIdx is the ID of the THREAD in the BLOCK
     *  - blockIdx  is the ID of the BLOCK in the GRID
     */
    #define THREADS_X (gridDim.x*blockDim.x)
    #define THREADS_Y (gridDim.y*blockDim.y)

    #define THREAD_ID_X (blockIdx.x*blockDim.x+threadIdx.x)
    #define THREAD_ID_Y (blockIdx.y*blockDim.y+threadIdx.y)

    #define BLOCKS_TOT (gridDim.x*gridDim.y)
    #define BLOCK_INDEX (blockIdx.y*gridDim.x+blockIdx.x)
#else
    #define KERNEL_FUNCTION(type, name) type cpu_##name

    #define THREADS_X 1
    #define THREADS_Y 1

    #define THREAD_ID_X 0
    #define THREAD_ID_Y 0

    #define BLOCKS_TOT 1
    #define BLOCK_INDEX 0
#endif

/* Non-arch-specific macros. */
#define THREAD_ID (THREADS_X*THREAD_ID_Y + THREAD_ID_X)
#define THREADS_TOT (THREADS_X*THREADS_Y)

#define ERROR(id, x, y, i) error_buffer[errbuf] = (id); \
    error_buffer[errbuf+1] = (x); \
    error_buffer[errbuf+2] = (y); \
    errbuf = (errbuf+(i)) % (MAX_ERRORS*3)

/* Profiling macros.
 *    _START() creates a timer and starts
 *    _POINT(str) prints "TIME ms, STR", time is since last _POINT or _START
 *    _END() stops the timer
 */
#ifdef KERNEL_PROFILER
    #define PROFILER_START QElapsedTimer PROFILER_TIMER; PROFILER_TIMER.start()
    #define PROFILER_POINT(str) qDebug("%lld ms, %s", PROFILER_TIMER.restart(), (str))
    #define PROFILER_END PROFILER_TIMER.invalidate()
#else
    #define PROFILER_START
    #define PROFILER_POINT
    #define PROFILER_END
#endif

/* The actual kernel function. */
KERNEL_FUNCTION(void, drc) (const char* pixels, int imgW, int imgH, int* error_buffer);

#endif // KERNEL_KERNEL_H

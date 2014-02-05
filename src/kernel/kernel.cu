#include "kernel.h"

#include <limits.h>
#include "errors.h"

#define ERROR(id, x, y) error_buffer[errbuf] = id; \
    error_buffer[errbuf+1] = x; \
    error_buffer[errbuf+2] = y; \
    errbuf += 3

KERNEL_FUNCTION(void, drc) (char* pixels, int imgW, int imgH, int* error_buffer)
{
    int totalThreads = THREADS_TOT, myThreadID = THREAD_ID;
    int maxErrors = (MAX_ERRORS/totalThreads);

    if(myThreadID > imgH || myThreadID > imgW)
        return;

    int errbuf = maxErrors*myThreadID;
    ERROR(I_THREAD_ID, myThreadID, THREADS_TOT);

#ifdef SMP
    SYNCTHREADS;
#endif

    /* Horizontal check */
    int y = myThreadID, pixelsSinceBlack;
    while(y < imgH) {
        pixelsSinceBlack = INT_MAX;
        for(int x = 0; x < imgW; x++) {
            if(pixels[imgW*y+x] == 'x') {
                if(pixelsSinceBlack < 4 && pixelsSinceBlack != 0) {
                    ERROR(E_HOR_SPACING_TOO_SMALL, x, y);
                }
                pixelsSinceBlack = 0;
            } else {
                pixelsSinceBlack++;
            }
        }
        y += totalThreads;
    }

#ifdef SMP
    SYNCTHREADS;
#endif

    /* Vertical check */
    int x = myThreadID;
    while(x < imgW) {
        pixelsSinceBlack = INT_MAX;
        for(int y = 0; y < imgH; y++) {
            if(pixels[imgW*y+x] == 'x') {
                if(pixelsSinceBlack < 4 && pixelsSinceBlack != 0) {
                    ERROR(E_VER_SPACING_TOO_SMALL, x, y);
                }
                pixelsSinceBlack = 0;
            } else {
                pixelsSinceBlack++;
            }
        }
        x += totalThreads;
    }
}

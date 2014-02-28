#include "kernel.h"

#include <limits.h>
#include "errors.h"
#include "design_rules.h"

#define BLACK_PIXEL 1

KERNEL_FUNCTION(void, drc) (const char* pixels, int imgW, int imgH, int* error_buffer)
{
    int totalThreads = THREADS_TOT, myThreadID = THREAD_ID;
    int maxErrors = (MAX_ERRORS/totalThreads);
    if(maxErrors == 0) { maxErrors = 1; }

    if(myThreadID > imgH || myThreadID > imgW)
        return;

    int errbuf = maxErrors*myThreadID*3;
    //ERROR(I_THREAD_ID, myThreadID, THREADS_TOT, 3);

#ifdef SMP
    SYNCTHREADS;
#endif

    /* Horizontal check */
    int y = myThreadID;
    int pixelsSinceFilled;
    while(y < imgH) {
        pixelsSinceFilled = INT_MAX-imgW;
        int rowbase = imgW*y;
        for(int x = 0; x < imgW; x++) {
            int isFilled = (pixels[rowbase+x] == BLACK_PIXEL);
            int increment = ((isFilled) && (pixelsSinceFilled < R_MIN_SPACE) && (pixelsSinceFilled != 0))
                    ? 3 : 0;
            ERROR(E_HOR_SPACING_TOO_SMALL, x, y, increment);
            pixelsSinceFilled = (pixelsSinceFilled+1)*(isFilled == 0);
        }
        y += totalThreads;
    }

#ifdef SMP
    SYNCTHREADS;
#endif

    /* Vertical check */
    int x = myThreadID;
    while(x < imgW) {
        pixelsSinceFilled = INT_MAX-imgH;
        for(int y = 0; y < imgH; y++) {
            int isFilled = (pixels[imgW*y+x] == BLACK_PIXEL);
            int increment = ((isFilled) && (pixelsSinceFilled < R_MIN_SPACE) && (pixelsSinceFilled != 0))
                    ? 3 : 0;
            ERROR(E_VER_SPACING_TOO_SMALL, x, y, increment);
            pixelsSinceFilled = (pixelsSinceFilled+1)*(isFilled == 0);
        }
        x += totalThreads;
    }

    ERROR(0, 0, 0, 0); // overwrite the last (invalid) error
}

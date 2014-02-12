#include "kernel.h"

#include <limits.h>
#include "errors.h"
#include "design_rules.h"

#include <QtGlobal>

#ifndef IFS
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
    int y = myThreadID, pixelsSinceBlack;
    while(y < imgH) {
        pixelsSinceBlack = INT_MAX;
        for(int x = 0; x < imgW; x++) {
            int isBlack = (pixels[imgW*y+x] == 'x');
            int error = (isBlack) && (pixelsSinceBlack < R_MIN_SPACE) && (pixelsSinceBlack != 0);
            ERROR(E_HOR_SPACING_TOO_SMALL*error, x*error, y*error, 3*error);
            pixelsSinceBlack = (pixelsSinceBlack+1)*(isBlack != 1);
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
            int isBlack = (pixels[imgW*y+x] == 'x');
            int error = (isBlack) && (pixelsSinceBlack < R_MIN_SPACE) && (pixelsSinceBlack != 0);
            ERROR(E_HOR_SPACING_TOO_SMALL*error, x*error, y*error, 3*error);
            pixelsSinceBlack = (pixelsSinceBlack+1)*(isBlack != 1);
        }
        x += totalThreads;
    }
}
#else
KERNEL_FUNCTION(void, drc) (const char* pixels, int imgW, int imgH, int* error_buffer)
{
    int totalThreads = THREADS_TOT, myThreadID = THREAD_ID;
    int maxErrors = (MAX_ERRORS/totalThreads);
    if(maxErrors == 0) { maxErrors = 1; }

    if(myThreadID > imgH || myThreadID > imgW)
        return;

    int errbuf = maxErrors*myThreadID*3;
    ERROR(I_THREAD_ID, myThreadID, THREADS_TOT, 3);

#ifdef SMP
    SYNCTHREADS;
#endif

    /* Horizontal check */
    int y = myThreadID, pixelsSinceBlack;
    while(y < imgH) {
        pixelsSinceBlack = INT_MAX;
        for(int x = 0; x < imgW; x++) {
            if(pixels[imgW*y+x] == 'x') {
                if(pixelsSinceBlack < R_MIN_SPACE && pixelsSinceBlack != 0) {
                    ERROR(E_HOR_SPACING_TOO_SMALL, x, y, 3);
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
                if(pixelsSinceBlack < R_MIN_SPACE && pixelsSinceBlack != 0) {
                    ERROR(E_VER_SPACING_TOO_SMALL, x, y, 3);
                }
                pixelsSinceBlack = 0;
            } else {
                pixelsSinceBlack++;
            }
        }
        x += totalThreads;
    }
}
#endif

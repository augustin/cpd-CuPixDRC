#include "kernel.h"

#include <limits.h>
#include "errors.h"

KERNEL_FUNCTION(void, drc) (char* pixels, int imgW, int imgH, int* error_buffer)
{
    int totalThreads = THREADS_TOT, myThreadID = THREAD_ID;
    int maxErrors = (MAX_ERRORS/totalThreads);

    if(myThreadID > imgH || myThreadID > imgW)
        return;

    int secW = imgW/totalThreads, secH = imgH/totalThreads;
    if(secW == 0) { secW = 1; }
    if(secH == 0) { secH = 1; }
    int mySecX = secW*myThreadID, mySecY = secH*myThreadID;

    int errbuf = maxErrors*myThreadID;
    error_buffer[errbuf] = I_THREAD_ID;
    error_buffer[errbuf+1] = myThreadID;
    error_buffer[errbuf+2] = THREADS_TOT;
    error_buffer[errbuf+3] = I_SEC_DIM;
    error_buffer[errbuf+4] = secW;
    error_buffer[errbuf+5] = secH;
    errbuf += 6;

#ifdef SMP
    SYNCTHREADS;
#endif

    int pixelsSinceBlack = INT_MAX;
    for(int y = mySecY; y < mySecY+secH; y++) {
		pixelsSinceBlack = INT_MAX;
        for(int x = mySecX; x < mySecX+secW; x++) {
			if(pixels[imgW*y+x] == 'x') {
				if(pixelsSinceBlack < 4 && pixelsSinceBlack != 0) {
					error_buffer[errbuf] = E_HOR_SPACING_TOO_SMALL;
					error_buffer[errbuf+1] = x;
					error_buffer[errbuf+2] = y;
					errbuf += 3;
				}
				pixelsSinceBlack = 0;
			} else {
				pixelsSinceBlack++;
			}
		}
	}

#ifdef SMP
    SYNCTHREADS;
#endif

    for(int x = mySecX; x < mySecX+secW; x++) {
		pixelsSinceBlack = INT_MAX;
        for(int y = mySecY; y < mySecY+secH; y++) {
			if(pixels[imgW*y+x] == 'x') {
				if(pixelsSinceBlack < 4 && pixelsSinceBlack != 0) {
					error_buffer[errbuf] = E_VER_SPACING_TOO_SMALL;
					error_buffer[errbuf+1] = x;
					error_buffer[errbuf+2] = y;
					errbuf += 3;
				}
				pixelsSinceBlack = 0;
			} else {
				pixelsSinceBlack++;
			}
		}
	}
}

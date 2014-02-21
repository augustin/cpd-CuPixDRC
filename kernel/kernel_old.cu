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
    int y = myThreadID, pixelsSinceFilled;
    while(y < imgH) {
        pixelsSinceFilled = INT_MAX;
        for(int x = 0; x < imgW; x++) {
            if(pixels[imgW*y+x] == 'x') {
                if(pixelsSinceFilled < R_MIN_SPACE && pixelsSinceFilled != 0) {
                    ERROR(E_HOR_SPACING_TOO_SMALL, x, y, 3);
                }
                pixelsSinceFilled = 0;
            } else {
                pixelsSinceFilled++;
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
        pixelsSinceFilled = INT_MAX;
        for(int y = 0; y < imgH; y++) {
            if(pixels[imgW*y+x] == 'x') {
                if(pixelsSinceFilled < R_MIN_SPACE && pixelsSinceFilled != 0) {
                    ERROR(E_VER_SPACING_TOO_SMALL, x, y, 3);
                }
                pixelsSinceFilled = 0;
            } else {
                pixelsSinceFilled++;
            }
        }
        x += totalThreads;
    }
}
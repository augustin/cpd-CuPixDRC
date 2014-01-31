#include "kernel.h"

#include <limits.h>
#include "errors.h"

KERNEL_FUNCTION(void, drc) (char* pixels, int imgW, int imgH, int* error_buffer)
{
	int pixelsSinceBlack = INT_MAX, errbuf = 0;

	for(int y = 0; y < 10; y++) {
		pixelsSinceBlack = INT_MAX;
		for(int x = 0; x < 10; x++) {
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

	for(int x = 0; x < 10; x++) {
		pixelsSinceBlack = INT_MAX;
		for(int y = 0; y < 10; y++) {
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

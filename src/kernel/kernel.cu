#include "kernel.h"

#include <limits.h>
#include <stdio.h>

KERNEL_FUNCTION(void, hello) (char* pixels[], int imgW, int imgH)
{
	int w = 10;
	char * testarray=
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
	int pixelsSinceBlack = INT_MAX;

	printf("starting horizontal check\n");
	for(int y = 0; y < 10; y++) {
		pixelsSinceBlack = INT_MAX;
		for(int x = 0; x < 10; x++) {
			if(testarray[w*y+x] == 'x') {
				if(pixelsSinceBlack < 4 && pixelsSinceBlack != 0) { printf("violation!! %d, %d\n", x, y); }
				pixelsSinceBlack = 0;
			}
			else { pixelsSinceBlack++; }
		}
	}

	printf("starting vertical check\n");
	for(int x = 0; x < 10; x++) {
		pixelsSinceBlack = INT_MAX;
		for(int y = 0; y < 10; y++) {
			if(testarray[w*y+x] == 'x') {
				if(pixelsSinceBlack < 4 && pixelsSinceBlack != 0) { printf("violation!! %d, %d\n", x, y); }
				pixelsSinceBlack = 0;
			}
			else { pixelsSinceBlack++; }
		}
	}

	printf("done!\n");
}

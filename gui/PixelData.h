#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdlib.h>

#define PIX(x, y, w) (y*w+x)

/* TODO!! */
struct PixelData {
	u_int8_t *pix;
	int32_t width, height;
};

#endif // GLOBALS_H

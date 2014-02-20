#ifndef KERNEL_KERNEL_H
#define KERNEL_KERNEL_H

#include "kernel_defs.h"

KERNEL_FUNCTION(void, drc) (const char* pixels, int imgW, int imgH, int* error_buffer);

#endif // KERNEL_KERNEL_H

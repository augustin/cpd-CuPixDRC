#include "kernel.h"

KERNEL_FUNCTION(void, hello) (char *a, int *b)
{
	a[0] += b[0];
}

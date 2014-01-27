#ifndef KERNEL_DEFS_H
#define KERNEL_DEFS_H

/* Some people say this is "abusing the preprocessor", but it should work
 * on any modern compiler. Two '#'s means print the literal, one '#' means
 * print the literal as a string. */
#if defined(CUDA)
#	define KERNEL_FUNCTION(type, name) __global__ type device_##name
#else
#	define KERNEL_FUNCTION(type, name) type cpu_##name
#endif

#endif // KERNEL_DEFS_H

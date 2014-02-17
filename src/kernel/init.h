#ifndef KERNEL_INIT_H
#define KERNEL_INIT_H

#define N 16
#define blocksize 16

int* kernel_main_cuda(int device, const char *pixels, int w, int h, int blocks, int threads);
int* kernel_main_cpu(const char* pixels, int w, int h);

#endif // KERNEL_INIT_H


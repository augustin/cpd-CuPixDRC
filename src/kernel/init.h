#ifndef KERNEL_INIT_H
#define KERNEL_INIT_H

#define N 16
#define blocksize 16

int* kernel_main_cuda(int device);
int* kernel_main_cpu();

#endif // KERNEL_INIT_H


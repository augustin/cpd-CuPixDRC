#ifndef KERNEL_INIT_H
#define KERNEL_INIT_H

#define N 16
#define blocksize 16

void kernel_main_cuda(int device);
void kernel_main_cpu();

#endif // KERNEL_INIT_H


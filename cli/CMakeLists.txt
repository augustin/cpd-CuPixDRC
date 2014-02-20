cmake_minimum_required(VERSION 2.8)
project(CuPixDRC_cli)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../CMakeTests)
find_package(AnyQt)
find_package(CUDA)
include_directories(${QT_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_KERNEL_SRCS ../kernel/init.cu
                     ../kernel/kernel.cu)

set(CPU_KERNEL_SRCS  ../kernel/init.cpp
                     ../kernel/kernel.cpp)

set(CUDA_KERNEL_HDRS ../kernel/kernel.h
                     ../kernel/init.h
                     ../kernel/errors.h
                     ../kernel/kernel_defs.h
                     ../kernel/design_rules.h)

set(SRCS Main.cpp)

add_executable(CpuPixDRC_cli ${CPU_KERNEL_SRCS} ${CUDA_KERNEL_HDRS} ${SRCS})
target_link_libraries(CpuPixDRC_cli ${QT_LIBS})

cuda_compile(CUDA_KERNEL_OBJS ${CUDA_KERNEL_SRCS} OPTIONS -DCUDA)
cuda_add_executable(CudaPixDRC_cli ${CUDA_KERNEL_OBJS} ${SRCS})
set_target_properties(CudaPixDRC_cli PROPERTIES COMPILE_DEFINITIONS CUDA)
target_link_libraries(CudaPixDRC_cli ${QT_LIBS})

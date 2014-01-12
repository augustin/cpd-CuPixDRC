cmake_minimum_required(VERSION 2.8)
project(CuPixDRC)

include(../CMakeTests/FindAnyQt.cmake)
include_directories(${QT_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA)
set(CUDA_KERNEL_SRCS kernel/kernel.cu
                     kernel/init.cu)
set(CUDA_KERNEL_HDRS kernel/kernel.h
                     kernel/init.h)

set(SRCS DRCwind.cpp
         widgets/ErrorList.cpp
         dialogs/SelectDevice.cpp
         Main.cpp)
set(HDRS DRCwind.h
         widgets/ErrorList.h
         dialogs/SelectDevice.h)
set(UIS DRCwind.ui
        widgets/ErrorList.ui
        dialogs/SelectDevice.ui)
qt_moc(MOC_SRCS ${HDRS})
qt_ui(UIC_HDRS ${UIS})

cuda_compile(CUDA_KERNEL_OBJS ${CUDA_KERNEL_SRCS})
cuda_add_executable(CuPixDRC ${CUDA_KERNEL_OBJS} ${UIC_HDRS} ${MOC_SRCS} ${SRCS})
target_link_libraries(CuPixDRC ${QT_LIBS})

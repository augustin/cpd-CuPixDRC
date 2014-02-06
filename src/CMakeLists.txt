cmake_minimum_required(VERSION 2.8)
project(CuPixDRC)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../CMakeTests)
find_package(AnyQt)
find_package(CUDA)
include_directories(${QT_INCLUDE} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(../../ChipLib/ChipLib "${CMAKE_CURRENT_BINARY_DIR}/ChipLib")
include_directories(../../ChipLib/ChipLib)
include_directories(../../ChipLib/ChipDisplay)

set(CUDA_KERNEL_SRCS kernel/init.cu
                     kernel/kernel.cu)

set(CPU_KERNEL_SRCS  kernel/init.cpp
                     kernel/kernel.cpp)

set(CUDA_KERNEL_HDRS kernel/kernel.h
                     kernel/init.h
                     kernel/errors.h
                     kernel/kernel_defs.h
                     kernel/design_rules.h)

set(SRCS DRCwind.cpp
         widgets/ErrorList.cpp
         Main.cpp
         ImageRequester.cpp)

set(HDRS DRCwind.h
         widgets/ErrorList.h)

qt_moc(MOC_SRCS ${HDRS})
qt_moc(MOC_CUDA_SRCS dialogs/SelectDevice.h)

set(UIS DRCwind.ui
        widgets/ErrorList.ui)
qt_ui(UIC_HDRS ${UIS})
qt_ui(UIC_CUDA_HDRS dialogs/SelectDevice.ui)

add_executable(CpuPixDRC ${CPU_KERNEL_SRCS} ${CUDA_KERNEL_HDRS}
                         ${UIC_HDRS} ${MOC_SRCS}
                         ${SRCS})
target_link_libraries(CpuPixDRC ChipLib ${QT_LIBS})

cuda_compile(CUDA_KERNEL_OBJS ${CUDA_KERNEL_SRCS} OPTIONS -DCUDA)
cuda_add_executable(CudaPixDRC ${CUDA_KERNEL_OBJS}
                               ${UIC_HDRS} ${MOC_SRCS}
                                    dialogs/SelectDevice.cpp ${UIC_CUDA_HDRS} ${MOC_CUDA_SRCS}
                               ${SRCS})
set_target_properties(CudaPixDRC PROPERTIES COMPILE_DEFINITIONS CUDA)
target_link_libraries(CudaPixDRC ChipLib ${QT_LIBS})

cmake_minimum_required(VERSION 3.8)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

project(stencil CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

add_executable(0_stencil 0_stencil.cpp)
add_executable(1_stencil_gpu 1_stencil_gpu.cu)
add_executable(2_stencil_tiling 2_stencil_tiling.cu)
add_executable(3_stencil_coarsening_slicing 3_stencil_coarsening_slicing.cu)
add_executable(4_stencil_register_tiling 4_stencil_register_tiling.cu)

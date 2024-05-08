cmake_minimum_required(VERSION 3.8)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

project(stencil CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

add_executable(0_convolution 0_convolution.cpp)
add_executable(1_convolution_gpu 1_convolution_gpu.cu)
add_executable(2_convolution_constant 2_convolution_constant.cu)
add_executable(3_convolution_constant_coarsening 3_convolution_constant_coarsening.cu)
add_executable(4_convolution_tiled 4_convolution_tiled.cu)
add_executable(5_convolution_tiled_cache 5_convolution_tiled_cache.cu)

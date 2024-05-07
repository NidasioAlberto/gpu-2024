cmake_minimum_required(VERSION 3.8)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

project(stencil CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
endif()

add_executable(1_convolution_naive 1_convolution_naive.cu)
add_executable(2_convolution_constant 2_convolution_constant.cu)
add_executable(3_convolution_tiled 3_convolution_tiled.cu)

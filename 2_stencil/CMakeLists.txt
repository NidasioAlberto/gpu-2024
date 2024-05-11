cmake_minimum_required(VERSION 3.8)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

project(stencil CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS_DEBUG ${CMAKE_CUDA_FLAGS} "-g -G")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")

add_executable(1_stencil_naive 1_stencil_naive.cu)
add_executable(2_stencil_tiling 2_stencil_tiling.cu)
add_executable(3_stencil_coarsening 3_stencil_coarsening.cu)
add_executable(4_stencil_registers 4_stencil_registers.cu)

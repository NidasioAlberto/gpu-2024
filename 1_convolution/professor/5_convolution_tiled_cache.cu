#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                         \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess) {                                                         \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

using input_type = float;
using filter_type = input_type;

#define FILTER_RADIUS 8
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)
__constant__ filter_type constant_filter[FILTER_SIZE][FILTER_SIZE];

#define TILE_DIM 32

#define THREADS_PER_AXIS TILE_DIM

void convolution_cpu(input_type *input, const input_type *filter, input_type *output, const int width, const int height,
                     const int filter_size, const int filter_radius) {
    for (int outRow = 0; outRow < width; outRow++) {
        for (int outCol = 0; outCol < height; outCol++) {
            input_type value{0.0f};
            for (int row = 0; row < filter_size; row++)
                for (int col = 0; col < filter_size; col++) {
                    int inRow = outRow - filter_radius + row;
                    int inCol = outCol - filter_radius + col;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        value += filter[row * filter_size + col] * input[inRow * width + inCol];
                    }
                }
            output[outRow * width + outCol] = value;
        }
    }
}

// GPU filter for convolution TILED
__global__ void convolution_tiled_kernel(const input_type *__restrict__ input, input_type *__restrict__ output,
                                         const int width, const int height) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int col = blockIdx.x * TILE_DIM + tidx;
    const int row = blockIdx.y * TILE_DIM + tidy;

    // Load input tile
    __shared__ input_type input_shared[TILE_DIM][TILE_DIM];
    if (row < height && col < width) {
        input_shared[tidy][tidx] = input[row * width + col];
    } else {
        input_shared[tidy][tidx] = 0.0;
    }
    __syncthreads();

    // Compute output elements
    if (col < width && row < height) {
        float PValue = 0.0f;
#pragma unroll
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
#pragma unroll
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                if (tidx - FILTER_RADIUS + fCol >= 0 && tidx - FILTER_RADIUS + fCol < TILE_DIM &&
                    tidy - FILTER_RADIUS + fRow >= 0 && tidy - FILTER_RADIUS + fRow < TILE_DIM) {
                    PValue += constant_filter[fRow][fCol] *
                              input_shared[tidy - FILTER_RADIUS + fRow][tidx - FILTER_RADIUS + fCol];
                } else {
                    if (row - FILTER_RADIUS + fRow >= 0 && row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 && col - FILTER_RADIUS + fCol < width) {
                        PValue += constant_filter[fRow][fCol] *
                                  input[(row - FILTER_RADIUS + fRow) * width + col - FILTER_RADIUS + fCol];
                    }
                }
            }
        }
        output[row * width + col] = PValue;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Please specify matrix dimensions\n");
        return EXIT_FAILURE;
    }
    const unsigned dim = atoi(argv[1]);
    assert(dim % TILE_DIM == 0);
    const unsigned int width = dim;
    const unsigned int height = dim;

    input_type *input = new input_type[width * height];                // Input
    filter_type *filter = new filter_type[FILTER_SIZE * FILTER_SIZE];  // Convolution filter
    input_type *output_cpu = new input_type[width * height];           // Output (CPU)
    input_type *output_gpu = new input_type[width * height];           // Output (GPU)

    // Randomly initialize the inputs
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) filter[i] = static_cast<filter_type>(rand()) / RAND_MAX;

    for (int i = 0; i < width * height; ++i)
        input[i] = static_cast<input_type>(rand()) / RAND_MAX;  // Random value between 0 and 1

    // Call CPU convolution
    convolution_cpu(input, filter, output_cpu, width, height, FILTER_SIZE, FILTER_RADIUS);

    // Allocate GPU memory and copy data
    input_type *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, width * height * sizeof(input_type)));
    CHECK(cudaMalloc(&d_output, width * height * sizeof(input_type)));
    CHECK(cudaMemcpy(d_input, input, width * height * sizeof(input_type), cudaMemcpyHostToDevice));

    //  Call TILED GPU convolution kernel
    //  Same (block, grid) as above
    //  Same constant memory as before
    const dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    const dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
    CHECK(cudaMemcpyToSymbol(constant_filter, filter, FILTER_SIZE * FILTER_SIZE * sizeof(filter_type)));
    convolution_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK(cudaMemcpy(output_gpu, d_output, width * height * sizeof(input_type), cudaMemcpyDeviceToHost));

    // Compare results (e.g., check if output_cpu and output_gpu are equivalent)
    for (int i = 0; i < width * height; i++) {
        if (std::abs(output_cpu[i] - output_gpu[i]) > 1e-3) {
            std::cout << "Convolution TILED results are NOT equivalent!" << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Convolution TILED results are equivalent!" << std::endl;

    // Cleanup and deallocate memory
    delete[] input;
    delete[] filter;
    delete[] output_cpu;
    delete[] output_gpu;
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}
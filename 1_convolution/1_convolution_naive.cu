#include <cuda_runtime.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

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

#define DATA_DIM 1024
#define DATA_SIZE (DATA_DIM * DATA_DIM)

#define FILTER_RADIUS 5
#define FILTER_DIM (FILTER_RADIUS * 2 + 1)
#define FILTER_SIZE (FILTER_DIM * FILTER_DIM)

#define BLOCK_DIM 16
#define GRID_DIM (DATA_DIM / BLOCK_DIM)

__global__ void convolution_naive(float *__restrict__ input, float *__restrict__ filter, float *__restrict__ output) {
    // Data position
    int x_data = blockIdx.x * blockDim.x + threadIdx.x;
    int y_data = blockIdx.y * blockDim.y + threadIdx.y;

    float value = 0.0f;

    for (int x_f = 0; x_f < FILTER_DIM; x_f++) {
        for (int y_f = 0; y_f < FILTER_DIM; y_f++) {
            // Global coordinates of current required cell
            int x_cell_global = x_data - FILTER_RADIUS + x_f;
            int y_cell_global = y_data - FILTER_RADIUS + y_f;

            // Check boundary conditions
            if (x_cell_global >= 0 && x_cell_global < DATA_DIM && y_cell_global >= 0 && y_cell_global < DATA_DIM) {
                value += filter[y_f * FILTER_DIM + x_f] * input[y_cell_global * DATA_DIM + x_cell_global];

                // One multiplication and one addition => 2 flops
                // One read from filter and one from input => 2 memory accesses => 8 bytes
                // 2 / 8 = 0.25
            }
        }
    }

    output[y_data * DATA_DIM + x_data] = value;
}

int main() {
    std::vector<float> input(DATA_SIZE);
    std::vector<float> filter(FILTER_SIZE);
    std::vector<float> output(DATA_SIZE);

    // Randomly initialize input and filter
    for (int i = 0; i < DATA_SIZE; i++) input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < FILTER_SIZE; i++) filter[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate GPU memory and copy data
    float *d_input, *d_filter, *d_output;
    CHECK(cudaMalloc(&d_input, DATA_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_filter, FILTER_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_output, DATA_SIZE * sizeof(float)));
    CHECK(cudaMemcpy(d_input, input.data(), DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter, filter.data(), FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Call GPU kernel
    const dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    const dim3 blocks_per_grid(GRID_DIM, GRID_DIM);
    convolution_naive<<<blocks_per_grid, threads_per_block>>>(d_input, d_filter, d_output);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK(cudaMemcpy(output.data(), d_output, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device resources
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_filter));
    CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}

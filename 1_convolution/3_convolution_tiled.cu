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

#define TILE_DIM 16

#define BLOCK_DIM TILE_DIM
#define GRID_DIM (DATA_DIM / BLOCK_DIM)

__constant__ float constant_filter[FILTER_DIM][FILTER_DIM];

#define THREADS_PER_AXIS 32

__global__ void convolution_tiled(float *__restrict__ input, float *__restrict__ output) {
    // Data position
    int x_data = blockIdx.x * blockDim.x + threadIdx.x;
    int y_data = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory: 48KB per block
    // Here we use: 16**2*4 = 1024B
    __shared__ float input_tile[TILE_DIM][TILE_DIM];

    // Load data into shared memory
    if (x_data >= 0 && x_data < DATA_DIM && y_data >= 0 && y_data < DATA_DIM) {
        input_tile[threadIdx.y][threadIdx.x] = input[y_data * DATA_DIM + x_data];
    } else {
        input_tile[threadIdx.y][threadIdx.x] = 0;
    }

    // Ensure that at this point the threads have all copied into shared memory the datum
    __syncthreads();

    float value = 0.0f;

    // #pragma unroll
    for (int x_f = 0; x_f <= FILTER_DIM; x_f++) {
        // #pragma unroll
        for (int y_f = 0; y_f <= FILTER_DIM; y_f++) {
            // Global coordinates of current required cell
            int x_cell_global = x_data - FILTER_RADIUS + x_f;
            int y_cell_global = y_data - FILTER_RADIUS + y_f;

            // Local coordinates of current required cell
            int x_cell_tile = threadIdx.x - FILTER_RADIUS + x_f;
            int y_cell_tile = threadIdx.y - FILTER_RADIUS + y_f;

            // Check if we need data out of the tile
            if (x_cell_tile >= 0 && x_cell_tile <= TILE_DIM && y_cell_tile >= 0 && y_cell_tile <= TILE_DIM) {
                value += constant_filter[y_f][x_f] * input_tile[y_cell_tile][x_cell_tile];
                // Check if we need data out of the boundaries
            } else if (x_cell_global >= 0 && x_cell_global < DATA_DIM && y_cell_global >= 0 &&
                       y_cell_global < DATA_DIM) {
                value += constant_filter[y_f][x_f] * input[y_cell_global * DATA_DIM + x_cell_global];
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
    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, DATA_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_output, DATA_SIZE * sizeof(float)));
    CHECK(cudaMemcpy(d_input, input.data(), DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(constant_filter, filter.data(), FILTER_SIZE * sizeof(float)));

    // Call GPU kernel
    const dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    const dim3 blocks_per_grid(GRID_DIM, GRID_DIM);
    convolution_tiled<<<blocks_per_grid, threads_per_block>>>(d_input, d_output);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CHECK(cudaMemcpy(output.data(), d_output, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device resources
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}

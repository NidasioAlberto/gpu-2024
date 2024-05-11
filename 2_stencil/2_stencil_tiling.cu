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

#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

#define DATA_DIM (512 + 256)
#define DATA_SIZE (DATA_DIM * DATA_DIM * DATA_DIM)

// Helper macro
#define idx(x, y, z, N) ((z) * N * N + (y) * N + (x))

// Coefficients
//           (-y)
//            c3  (+z)
//            |  c6
//            | /
// (-x) c1----c0----c2 (+x)
//          / |
//        c5  |
//     (-z)   c4
//           (+y)
struct StencilCoefficients {
    float c0 = 1.0;  // Center
    float c1 = 1.0;  // x - 1
    float c2 = 1.0;  // x + 1
    float c3 = 1.0;  // y - 1
    float c4 = 1.0;  // y + 1
    float c5 = 1.0;  // z - 1
    float c6 = 1.0;  // z + 1
};

StencilCoefficients coefficients;
__constant__ StencilCoefficients constant_coefficients;

__global__ void stencil_kernel_gpu(const float *__restrict__ input, float *__restrict__ output) {
    const unsigned int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    const unsigned int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    const unsigned int z = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;

    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if (x < DATA_DIM && y < DATA_DIM && z < DATA_DIM) {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[idx(x, y, z, DATA_DIM)];
    }
    __syncthreads();

    if (x >= 1 && x < DATA_DIM - 1 && y >= 1 && y < DATA_DIM - 1 && z >= 1 && z < DATA_DIM - 1) {
        // Exclude halo cells
        if (threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1 && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1) {
            float value = constant_coefficients.c0 * tile[threadIdx.z][threadIdx.y][threadIdx.x];
            value += constant_coefficients.c1 * tile[threadIdx.z][threadIdx.y][threadIdx.x - 1];
            value += constant_coefficients.c2 * tile[threadIdx.z][threadIdx.y][threadIdx.x + 1];
            value += constant_coefficients.c3 * tile[threadIdx.z][threadIdx.y - 1][threadIdx.x];
            value += constant_coefficients.c4 * tile[threadIdx.z][threadIdx.y + 1][threadIdx.x];
            value += constant_coefficients.c5 * tile[threadIdx.z - 1][threadIdx.y][threadIdx.x];
            value += constant_coefficients.c6 * tile[threadIdx.z + 1][threadIdx.y][threadIdx.x];

            output[idx(x, y, z, DATA_DIM)] = value;
        }
    }
}

void stencil_cpu(const float *input, float *output) {
    for (int x = 1; x < DATA_DIM - 1; x++) {
        for (int y = 1; y < DATA_DIM - 1; y++) {
            for (int z = 1; z < DATA_DIM - 1; z++) {
                int output_idx = idx(x, y, z, DATA_DIM);

                float value = coefficients.c0 * input[output_idx];
                value += coefficients.c1 * input[idx(x - 1, y, z, DATA_DIM)];
                value += coefficients.c2 * input[idx(x + 1, y, z, DATA_DIM)];
                value += coefficients.c3 * input[idx(x, y - 1, z, DATA_DIM)];
                value += coefficients.c4 * input[idx(x, y + 1, z, DATA_DIM)];
                value += coefficients.c5 * input[idx(x, y, z - 1, DATA_DIM)];
                value += coefficients.c6 * input[idx(x, y, z + 1, DATA_DIM)];

                output[output_idx] = value;
            }
        }
    }
}

int main() {
    static_assert(DATA_DIM % BLOCK_DIM == 0,
                  "The dimension should be divisible by the number of output block computer by each block");

    std::vector<float> input(DATA_SIZE);

    // Randomly initialize input and coefficients
    for (int i = 0; i < DATA_SIZE; ++i) input[i] = static_cast<float>(rand()) / RAND_MAX;
    coefficients.c0 = static_cast<float>(rand()) / RAND_MAX;
    coefficients.c1 = static_cast<float>(rand()) / RAND_MAX;
    coefficients.c2 = static_cast<float>(rand()) / RAND_MAX;
    coefficients.c3 = static_cast<float>(rand()) / RAND_MAX;
    coefficients.c4 = static_cast<float>(rand()) / RAND_MAX;
    coefficients.c5 = static_cast<float>(rand()) / RAND_MAX;
    coefficients.c6 = static_cast<float>(rand()) / RAND_MAX;

    // Compute stencil on CPU
    std::vector<float> cpu_result(DATA_SIZE);
    stencil_cpu(input.data(), cpu_result.data());

    // Allocate GPU memory and copy data
    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, DATA_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&d_output, DATA_SIZE * sizeof(float)));
    CHECK(cudaMemcpy(d_input, input.data(), DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(constant_coefficients, &coefficients, sizeof(StencilCoefficients)));

    // Call GPU kernel
    dim3 threads_per_block(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 blocks_per_grid(DATA_DIM / OUT_TILE_DIM, DATA_DIM / OUT_TILE_DIM, DATA_DIM / OUT_TILE_DIM);
    stencil_kernel_gpu<<<blocks_per_grid, threads_per_block>>>(d_input, d_output);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<float> gpu_result(DATA_SIZE);
    CHECK(cudaMemcpy(gpu_result.data(), d_output, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare CPU and GPU results
    for (int x = 1; x < DATA_DIM - 1; x++) {
        for (int y = 1; y < DATA_DIM - 1; y++) {
            for (int z = 1; z < DATA_DIM - 1; z++) {
                int i = idx(x, y, z, DATA_DIM);

                if (cpu_result[i] != gpu_result[i]) {
                    std::cout << "Stencil CPU and GPU are NOT equivalent!" << std::endl;
                    std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;
                    std::cout << "i: " << i << std::endl;
                    std::cout << "CPU: " << cpu_result[i] << std::endl;
                    std::cout << "GPU: " << gpu_result[i] << std::endl;
                    return EXIT_FAILURE;
                }
            }
        }
    }
    for (int i = 0; i < DATA_SIZE; ++i) {
    }
    std::cout << "Stencil CPU and GPU are equivalent!" << std::endl;

    // Free memory
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}

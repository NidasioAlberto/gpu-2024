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
    float c0 = 1;  // Center
    float c1 = 1;  // x - 1
    float c2 = 1;  // x + 1
    float c3 = 1;  // y - 1
    float c4 = 1;  // y + 1
    float c5 = 1;  // z - 1
    float c6 = 1;  // z + 1
};

StencilCoefficients coefficients;
__constant__ StencilCoefficients constant_coefficients;

__global__ void stencil_kernel_gpu(const float *__restrict__ in, float *__restrict__ out) {
    const unsigned int x = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not compute the boundary portion of the matrix
    if (x >= 1 && x < DATA_DIM - 1 && y >= 1 && y < DATA_DIM - 1 && z >= 1 && z < DATA_DIM - 1) {
        int out_idx = idx(x, y, z, DATA_DIM);

        float value = constant_coefficients.c0 * in[out_idx];
        value += constant_coefficients.c1 * in[idx(x - 1, y, z, DATA_DIM)];
        value += constant_coefficients.c2 * in[idx(x + 1, y, z, DATA_DIM)];
        value += constant_coefficients.c3 * in[idx(x, y - 1, z, DATA_DIM)];
        value += constant_coefficients.c4 * in[idx(x, y + 1, z, DATA_DIM)];
        value += constant_coefficients.c5 * in[idx(x, y, z - 1, DATA_DIM)];
        value += constant_coefficients.c6 * in[idx(x, y, z + 1, DATA_DIM)];

        out[out_idx] = value;

        // 13 floating point operations
        // 7 reads from global memory -> 28B
        // 1 write -> 4B
    }
}

void stencil_cpu(const float *in, float *out) {
    for (int x = 1; x < DATA_DIM - 1; x++)
        for (int y = 1; y < DATA_DIM - 1; y++)
            for (int z = 1; z < DATA_DIM - 1; z++) {
                int out_idx = idx(x, y, z, DATA_DIM);

                float value = coefficients.c0 * in[out_idx];
                value += coefficients.c1 * in[idx(x - 1, y, z, DATA_DIM)];
                value += coefficients.c2 * in[idx(x + 1, y, z, DATA_DIM)];
                value += coefficients.c3 * in[idx(x, y - 1, z, DATA_DIM)];
                value += coefficients.c4 * in[idx(x, y + 1, z, DATA_DIM)];
                value += coefficients.c5 * in[idx(x, y, z - 1, DATA_DIM)];
                value += coefficients.c6 * in[idx(x, y, z + 1, DATA_DIM)];

                out[out_idx] = value;
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
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(DATA_DIM / BLOCK_DIM, DATA_DIM / BLOCK_DIM, DATA_DIM / BLOCK_DIM);
    stencil_kernel_gpu<<<blocks_per_grid, threads_per_block>>>(d_input, d_output);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<float> gpu_result(DATA_SIZE);
    CHECK(cudaMemcpy(gpu_result.data(), d_output, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare CPU and GPU results
    for (int i = 0; i < DATA_SIZE; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            std::cout << "Stencil CPU and GPU are NOT equivalent!" << std::endl;
            std::cout << "Index: " << i << std::endl;
            std::cout << "CPU: " << cpu_result[i] << std::endl;
            std::cout << "GPU: " << gpu_result[i] << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Stencil CPU and GPU are equivalent!" << std::endl;

    // Free memory
    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}

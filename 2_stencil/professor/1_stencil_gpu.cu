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

#define DIM (512 + 256)
#define SIZE (DIM * DIM * DIM)

#define get(data, i, j, k, N) data[(i) * N * N + (j) * N + (k)]

#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1

// Kernel function for GPU
__global__ void stencil_kernel_gpu(const float *__restrict__ in, float *__restrict__ out, const int N) {
    const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        get(out, i, j, k, N) = c0 * get(in, i, j, k, N) + c1 * get(in, i, j, k - 1, N) + c2 * get(in, i, j, k + 1, N) +
                               c3 * get(in, i, j - 1, k, N) + c4 * get(in, i, j + 1, k, N) +
                               c5 * get(in, i - 1, j, k, N) + c6 * get(in, i + 1, j, k, N);
    }
}

// Function for CPU stencil computation
void stencil_cpu(const float *in, float *out, const int N) {
    for (int i = 1; i < N - 1; ++i)
        for (int j = 1; j < N - 1; ++j)
            for (int k = 1; k < N - 1; ++k)
                get(out, i, j, k, N) = c0 * get(in, i, j, k, N) + c1 * get(in, i, j, k - 1, N) +
                                       c2 * get(in, i, j, k + 1, N) + c3 * get(in, i, j - 1, k, N) +
                                       c4 * get(in, i, j + 1, k, N) + c5 * get(in, i - 1, j, k, N) +
                                       c6 * get(in, i + 1, j, k, N);
}

int main() {
    static_assert(DIM / BLOCK_DIM,
                  "The dimension should be divisible by the number of output block computer by each block");
    // Generate random input data
    std::vector<float> input_data(SIZE);
    for (int i = 0; i < SIZE; ++i) {
        input_data[i] = rand() % 10;
    }

    // Compute stencil on CPU
    std::vector<float> cpu_result(SIZE);
    stencil_cpu(input_data.data(), cpu_result.data(), DIM);

    // Allocate memory on GPU
    float *input_data_gpu, *output_data_gpu;
    CHECK(cudaMalloc(&input_data_gpu, SIZE * sizeof(float)));
    CHECK(cudaMalloc(&output_data_gpu, SIZE * sizeof(float)));
    CHECK(cudaMemcpy(input_data_gpu, input_data.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Configure GPU kernel launch
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid((DIM + threads_per_block.x - 1) / threads_per_block.x,
                         (DIM + threads_per_block.y - 1) / threads_per_block.y,
                         (DIM + threads_per_block.z - 1) / threads_per_block.z);

    // Launch GPU kernel
    stencil_kernel_gpu<<<blocks_per_grid, threads_per_block>>>(input_data_gpu, output_data_gpu, DIM);
    CHECK_KERNELCALL();
    CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    std::vector<float> gpu_result(SIZE);
    CHECK(cudaMemcpy(gpu_result.data(), output_data_gpu, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare CPU and GPU results
    for (int i = 0; i < SIZE; ++i)
        if (cpu_result[i] != gpu_result[i]) {
            std::cout << "Stencil CPU and GPU are NOT equivalent!" << std::endl;
            std::cout << "Index: " << i << std::endl;
            std::cout << "CPU: " << cpu_result[i] << std::endl;
            std::cout << "GPU: " << gpu_result[i] << std::endl;
            return EXIT_FAILURE;
        }

    std::cout << "Stencil CPU and GPU are equivalent!" << std::endl;

    // Free memory
    CHECK(cudaFree(input_data_gpu));
    CHECK(cudaFree(output_data_gpu));

    return EXIT_SUCCESS;
}

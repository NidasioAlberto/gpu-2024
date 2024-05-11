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

#define Z_SLICING 32

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
__global__ void stencil_kernel_register_tiling_gpu(const float *__restrict__ in, float *__restrict__ out, const int N) {
    const unsigned int i_start = blockIdx.z * Z_SLICING;
    const unsigned int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    const unsigned int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    float in_prev;
    __shared__ float in_curr_s[IN_TILE_DIM][IN_TILE_DIM];
    float in_next;
    // Check greater than not needed since index is an unsigned int
    if (i_start - 1 < N && j < N && k < N) {
        in_prev = get(in, i_start - 1, j, k, N);
    }
    if (i_start < N && j < N && k < N) {
        in_curr_s[threadIdx.y][threadIdx.x] = get(in, i_start, j, k, N);
    }
    for (unsigned int i = i_start; i < i_start + Z_SLICING; ++i) {
        // Check greater than not needed since index is an unsigned int
        if (i + 1 < N && j < N && k < N) {
            in_next = get(in, i + 1, j, k, N);
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
                threadIdx.x < IN_TILE_DIM - 1) {
                get(out, i, j, k, N) =
                    c0 * in_curr_s[threadIdx.y][threadIdx.x] + c1 * in_curr_s[threadIdx.y][threadIdx.x - 1] +
                    c2 * in_curr_s[threadIdx.y][threadIdx.x + 1] + c3 * in_curr_s[threadIdx.y - 1][threadIdx.x] +
                    c4 * in_curr_s[threadIdx.y + 1][threadIdx.x] + c5 * in_prev + c6 * in_next;
            }
        }
        __syncthreads();
        in_prev = in_curr_s[threadIdx.y][threadIdx.x];
        in_curr_s[threadIdx.y][threadIdx.x] = in_next;
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
    static_assert(DIM / Z_SLICING, "The dimension should be divisible by the Z slicing");
    static_assert(DIM / OUT_TILE_DIM,
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
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid((DIM + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (DIM + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                         (DIM + Z_SLICING - 1) / Z_SLICING);

    // Launch GPU kernel
    stencil_kernel_register_tiling_gpu<<<blocks_per_grid, threads_per_block>>>(input_data_gpu, output_data_gpu, DIM);
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

#include <cuda_runtime.h>
#define N (1 << 10)

__global__ void vector_add(int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    int h_va[N], h_vb[N], h_vc[N];
    int *d_va, *d_vb, *d_vc;

    // Data initialization
    for (int i = 0; i < N; i++)
    {
        h_va[i] = i % 6;
        h_vb[i] = i % 8;
    }

    // Device memory allocation
    cudaMalloc(&d_va, N * sizeof(int));
    cudaMalloc(&d_vb, N * sizeof(int));
    cudaMalloc(&d_vc, N * sizeof(int));

    // CPU -> GPU data transmission
    cudaMemcpy(d_va, h_va, N * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_vb, h_vb, N * sizeof(int),
               cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 blocksPerGrid(N / 256, 1, 1);
    dim3 threadsPerBlock(256, 1, 1);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_vb,
                                                   d_vc);

    // GPU->CPU data transmission
    cudaMemcpy(h_vc, d_vc, N * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Device memory freeing
    cudaFree(d_va);
    cudaFree(d_vb);
    cudaFree(d_vc);

    return 0;
}

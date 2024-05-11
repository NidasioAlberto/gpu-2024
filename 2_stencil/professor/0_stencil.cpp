#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#define DIM 10
#define SIZE (DIM * DIM * DIM)

#define out(i, j, k) out[(i) * N * N + (j) * N + (k)]
#define in(i, j, k) in[(i) * N * N + (j) * N + (k)]

#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1

// Function for CPU stencil computation
void stencil_cpu(const float *in, float *out, const int N) {
    for (int i = 1; i < N - 1; ++i)
        for (int j = 1; j < N - 1; ++j)
            for (int k = 1; k < N - 1; ++k)
                out(i, j, k) = c0 * in(i, j, k) + c1 * in(i, j, k - 1) + c2 * in(i, j, k + 1) + c3 * in(i, j - 1, k) +
                               c4 * in(i, j + 1, k) + c5 * in(i - 1, j, k) + c6 * in(i + 1, j, k);
}

int main() {
    // Generate random input data
    std::vector<float> input_data(SIZE);
    for (int i = 0; i < SIZE; ++i) {
        input_data[i] = rand() % 10;
    }

    // Compute stencil on CPU
    std::vector<float> cpu_result(SIZE);
    stencil_cpu(input_data.data(), cpu_result.data(), DIM);

    return EXIT_SUCCESS;
}

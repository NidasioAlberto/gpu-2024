#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

using input_type = float;
using filter_type = input_type;

#define FILTER_RADIUS 4
#define FILTER_SIZE (FILTER_RADIUS * 2 + 1)

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

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Please specify matrix dimensions\n");
        return EXIT_FAILURE;
    }
    const unsigned dim = atoi(argv[1]);
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

    // Cleanup and deallocate memory
    delete[] input;
    delete[] filter;
    delete[] output_cpu;
    delete[] output_gpu;

    return EXIT_SUCCESS;
}
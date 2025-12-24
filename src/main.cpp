/**
 * @file main.cpp
 * @brief Minimal host driver that loads an image, runs the CUDA convolution,
 *        and writes the result to disk.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern "C" cudaError_t launchConvolution(const unsigned char* d_in, unsigned char* d_out, int w, int h, int f);

// Simple CUDA error macro for consistent diagnostics.
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t _err = (call);                                                              \
        if (_err != cudaSuccess) {                                                              \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));  \
        }                                                                                       \
    } while (0)

namespace {

constexpr const char* kOutputPath = "output.png";

/**
 * @brief Load an image from disk, run the CUDA convolution, and write output.
 */
void processImage(const std::string& inputPath, int filter) {
    int w, h, c;
    unsigned char* hostData = stbi_load(inputPath.c_str(), &w, &h, &c, 3);
    if (!hostData) {
        std::cerr << "Error: Could not load " << inputPath << "\n";
        return;
    }

    size_t size = w * h * 3;
    unsigned char *d_in, *d_out;
    std::vector<unsigned char> result(size);

    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, hostData, size, cudaMemcpyHostToDevice));

    CUDA_CHECK(launchConvolution(d_in, d_out, w, h, filter));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), d_out, size, cudaMemcpyDeviceToHost));

    stbi_write_png(kOutputPath, w, h, 3, result.data(), w * 3);
    std::cout << "SUCCESS: Saved to " << kOutputPath << "\n";

    cudaFree(d_in); cudaFree(d_out); stbi_image_free(hostData);
}

} // namespace

int main() {
    std::string filename;
    int filter = 0;
    bool running = true;
    int choice;

    while (running) {
        // Minimal menu for parsing
        if (!(std::cin >> choice)) break;

        switch (choice) {
            case 1: std::cin >> filename; break;
            case 2: {
                int inputFilter = 0;
                std::cin >> inputFilter;
                filter = std::clamp(inputFilter, 0, 3);
                break;
            }
            case 3:
                if (!filename.empty()) {
                    processImage(filename, filter);
                } else {
                    std::cerr << "No filename set.\n";
                }
                break;
            case 4: running = false; break;
        }
    }
    return 0;
}
/**
 * @file kernels.cu
 * @brief CUDA 2D convolution with shared-memory tiling and halo exchange.
 *
 * Each 16x16 thread block cooperatively loads an 18x18 tile (block + 1-pixel halo)
 * into shared memory. Boundary handling clamps coordinates so halo reads stay
 * in-bounds, removing branchy edge cases in the inner convolution loop. Filters
 * are stored in constant memory for fast broadcast to the SM.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

namespace {
constexpr int kMaskWidth = 3;
constexpr int kRadius = kMaskWidth / 2;
constexpr int kBlockDim = 16;

enum class FilterType : int {
    Identity = 0,
    GaussianBlur = 1,
    EdgeDetect = 2,
    Sharpen = 3
};

// Constant memory masks
__constant__ float c_identity[9] = { 0,0,0, 0,1,0, 0,0,0 };
__constant__ float c_gaussian[9] = { 1.f/16, 2.f/16, 1.f/16, 2.f/16, 4.f/16, 2.f/16, 1.f/16, 2.f/16, 1.f/16 };
__constant__ float c_edge[9]     = { 0,-1,0, -1,4,-1, 0,-1,0 };
__constant__ float c_sharpen[9]  = { 0,-1,0, -1,5,-1, 0,-1,0 };

__device__ __forceinline__ float maskValue(FilterType type, int idx) {
    switch (type) {
        case FilterType::GaussianBlur: return c_gaussian[idx];
        case FilterType::EdgeDetect: return c_edge[idx];
        case FilterType::Sharpen: return c_sharpen[idx];
        default: return c_identity[idx];
    }
}

__device__ __forceinline__ int clampCoord(int v, int limit) {
    return max(0, min(v, limit - 1));
}

/**
 * @brief Convolve an RGB image with a 3x3 filter using shared-memory tiling.
 *
 * A (16x16) tile of pixels plus a one-pixel halo on each side is loaded into
 * shared memory as a 18x18 tile. Threads collaboratively load center pixels,
 * edges, and corners to populate the halo. Coordinates are clamped when
 * sampling outside image bounds so the inner convolution loop can run without
 * additional boundary checks.
 */
__global__ void convolveRGB(const unsigned char* __restrict__ input,
                            unsigned char* __restrict__ output,
                            int width, int height, FilterType filter) {
    // Shared memory: block + halo (16 + 2*1 = 18 in each dimension)
    __shared__ float tile[kBlockDim + 2 * kRadius][kBlockDim + 2 * kRadius][3];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * kBlockDim + tx;
    const int y = blockIdx.y * kBlockDim + ty;

    const int sX = tx + kRadius;
    const int sY = ty + kRadius;

    // Helper to load global pixel to shared mem with boundary clamping
    auto load = [&](int gx, int gy, int sx, int sy) {
        int cx = clampCoord(gx, width);
        int cy = clampCoord(gy, height);
        int idx = (cy * width + cx) * 3;
        tile[sy][sx][0] = (float)input[idx];
        tile[sy][sx][1] = (float)input[idx + 1];
        tile[sy][sx][2] = (float)input[idx + 2];
    };

    // 1. Load center pixel
    load(x, y, sX, sY);

    // 2. Load Halos
    // Left & Right
    if (tx < kRadius) {
        load(x - kRadius, y, sX - kRadius, sY);
        load(x + kBlockDim, y, sX + kBlockDim, sY);
    }
    // Top & Bottom
    if (ty < kRadius) {
        load(x, y - kRadius, sX, sY - kRadius);
        load(x, y + kBlockDim, sX, sY + kBlockDim);
    }
    // Corners
    if (tx < kRadius && ty < kRadius) {
        load(x - kRadius, y - kRadius, sX - kRadius, sY - kRadius);
        load(x + kBlockDim, y - kRadius, sX + kBlockDim, sY - kRadius);
        load(x - kRadius, y + kBlockDim, sX - kRadius, sY + kBlockDim);
        load(x + kBlockDim, y + kBlockDim, sX + kBlockDim, sY + kBlockDim);
    }

    __syncthreads();

    // 3. Convolve
    if (x < width && y < height) {
        float r = 0.f;
        float g = 0.f;
        float b = 0.f;
        for (int ky = -kRadius; ky <= kRadius; ++ky) {
            for (int kx = -kRadius; kx <= kRadius; ++kx) {
                float w = maskValue(filter, (ky + kRadius) * kMaskWidth + (kx + kRadius));
                r += tile[sY + ky][sX + kx][0] * w;
                g += tile[sY + ky][sX + kx][1] * w;
                b += tile[sY + ky][sX + kx][2] * w;
            }
        }
        int outIdx = (y * width + x) * 3;
        output[outIdx]   = (unsigned char)fminf(fmaxf(r, 0.f), 255.f);
        output[outIdx+1] = (unsigned char)fminf(fmaxf(g, 0.f), 255.f);
        output[outIdx+2] = (unsigned char)fminf(fmaxf(b, 0.f), 255.f);
    }
}
} // namespace

/**
 * @brief Host-facing wrapper to launch the RGB convolution kernel.
 * @param d_input Device pointer to RGB input buffer (width * height * 3 bytes).
 * @param d_output Device pointer to RGB output buffer (same size as input).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param filterType Integer matching FilterType enum.
 */
extern "C" cudaError_t launchConvolution(const unsigned char* d_input, unsigned char* d_output,
                                         int width, int height, int filterType) {
    dim3 block(kBlockDim, kBlockDim);
    dim3 grid((width + kBlockDim - 1) / kBlockDim, (height + kBlockDim - 1) / kBlockDim);
    convolveRGB<<<grid, block>>>(d_input, d_output, width, height, static_cast<FilterType>(filterType));
    return cudaGetLastError();
}

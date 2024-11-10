#include "Compute.hpp"
#include "Image.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t error = call;                                                 \
        if (error != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

struct Lab {
    float L;
    float a;
    float b;
};

// Device functions for color space conversion and distance calculation
__device__ float sRGBToLinearGPU(float c) {
    return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
}

__device__ void rgbToXyzGPU(const rgb8& rgb, float& X, float& Y, float& Z) {
    float r = sRGBToLinearGPU(rgb.r / 255.0f);
    float g = sRGBToLinearGPU(rgb.g / 255.0f);
    float b = sRGBToLinearGPU(rgb.b / 255.0f);

    X = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    Y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    Z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
}

__device__ Lab xyzToLabGPU(float X, float Y, float Z) {
    const float Xr = 0.95047f;
    const float Yr = 1.00000f;
    const float Zr = 1.08883f;

    float x = X / Xr;
    float y = Y / Yr;
    float z = Z / Zr;

    float fx = (x > 0.008856f) ? cbrtf(x) : ((903.3f * x + 16.0f) / 116.0f);
    float fy = (y > 0.008856f) ? cbrtf(y) : ((903.3f * y + 16.0f) / 116.0f);
    float fz = (z > 0.008856f) ? cbrtf(z) : ((903.3f * z + 16.0f) / 116.0f);

    Lab lab;
    lab.L = 116.0f * fy - 16.0f;
    lab.a = 500.0f * (fx - fy);
    lab.b = 200.0f * (fy - fz);

    return lab;
}

__device__ Lab rgbToLabGPU(const rgb8& rgb) {
    float X, Y, Z;
    rgbToXyzGPU(rgb, X, Y, Z);
    return xyzToLabGPU(X, Y, Z);
}

__device__ float deltaEGPU(const Lab& lab1, const Lab& lab2) {
    float dL = lab1.L - lab2.L;
    float da = lab1.a - lab2.a;
    float db = lab1.b - lab2.b;
    return sqrtf(dL * dL + da * da + db * db);
}

// Kernel to compute distances between two images using Lab color space
__global__ void computeDistancesLabGPU(const rgb8* img1_buffer, const rgb8* img2_buffer, float* distances, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    rgb8 pixel1 = img1_buffer[idx];
    rgb8 pixel2 = img2_buffer[idx];

    Lab lab1 = rgbToLabGPU(pixel1);
    Lab lab2 = rgbToLabGPU(pixel2);
    distances[idx] = deltaEGPU(lab1, lab2);
}

// Kernel to apply a smooth filter based on distances
__global__ void applySmoothFilterCUDA(rgb8* in_buffer, float* distances, float min_threshold, float max_threshold, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float distance = distances[idx];

    // Smooth threshold range to reduce noise artifacts
    if (distance < min_threshold) {
        in_buffer[idx] = {0, 0, 0};  // Background
    } else if (distance > max_threshold) {
        in_buffer[idx] = {255, 0, 0};  // Foreground object in white
    } else {
        // Linear interpolation for smoother transition
        uint8_t intensity = static_cast<uint8_t>(255 * (distance - min_threshold) / (max_threshold - min_threshold));
        in_buffer[idx] = {intensity, intensity, intensity}; // Gray transition
    }
}

// Kernel to update the background image
__global__ void updateBackgroundCUDA(rgb8* bg_buffer, const rgb8* in_buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    rgb8 bg_pixel = bg_buffer[idx];
    rgb8 in_pixel = in_buffer[idx];

    bg_pixel.r = (bg_pixel.r + in_pixel.r) / 2;
    bg_pixel.g = (bg_pixel.g + in_pixel.g) / 2;
    bg_pixel.b = (bg_pixel.b + in_pixel.b) / 2;

    bg_buffer[idx] = bg_pixel;
}

void compute_cu(ImageView<rgb8> in) {
    static bool initialized = false;
    static Image<rgb8> device_bg;
    static Image<rgb8> device_candidate;
    static int time_since_match = 0;

    // Define thresholds for smoother filtering
    const float min_threshold = 10.0f;
    const float max_threshold = 20.0f;
    const int max_time_since_match = 50;

    // Allocate device memory for images and distances
    Image<rgb8> device_in(in.width, in.height, true);
    if (!initialized) {
        device_bg = Image<rgb8>(in.width, in.height, true);
        device_candidate = Image<rgb8>(in.width, in.height, true);
    }
    float* distances;
    CUDA_CHECK(cudaMalloc(&distances, in.width * in.height * sizeof(float)));

    // Copy input image to device
    CUDA_CHECK(cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride,
                            in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice));

    if (!initialized) {
        // Initialize background and candidate images
        CUDA_CHECK(cudaMemcpy(device_bg.buffer, device_in.buffer, in.width * in.height * sizeof(rgb8), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(device_candidate.buffer, device_in.buffer, in.width * in.height * sizeof(rgb8), cudaMemcpyDeviceToDevice));
        initialized = true;
    }

    // Compute distances between background and input image
    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    computeDistancesLabGPU<<<grid, block>>>(device_bg.buffer, device_in.buffer, distances, in.width, in.height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy distances to host to compute match_distance
    std::vector<float> host_distances(in.width * in.height);
    CUDA_CHECK(cudaMemcpy(host_distances.data(), distances, in.width * in.height * sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate match distances and apply background estimation logic
    double sum = 0.0;
    for (float d : host_distances) sum += d;
    double match_distance = sum / (in.width * in.height);

    // Background and candidate update logic
    if (match_distance < min_threshold) {
        // Update background on the device
        updateBackgroundCUDA<<<grid, block>>>(device_bg.buffer, device_in.buffer, in.width, in.height);
        CUDA_CHECK(cudaDeviceSynchronize());
        time_since_match = 0;
    } else if (time_since_match < max_time_since_match) {
        // Update candidate on the device
        updateBackgroundCUDA<<<grid, block>>>(device_candidate.buffer, device_in.buffer, in.width, in.height);
        CUDA_CHECK(cudaDeviceSynchronize());
        time_since_match++;
    } else {
        // Replace background with candidate
        CUDA_CHECK(cudaMemcpy(device_bg.buffer, device_candidate.buffer, in.width * in.height * sizeof(rgb8), cudaMemcpyDeviceToDevice));
        time_since_match = 0;
    }

    // Apply smooth threshold filter to reduce artifacts
    applySmoothFilterCUDA<<<grid, block>>>(device_in.buffer, distances, min_threshold, max_threshold, in.width, in.height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the filtered image back to the host
    CUDA_CHECK(cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride,
                            in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost));

    // Free device memory for distances
    CUDA_CHECK(cudaFree(distances));
}

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

template <typename T>
__device__ void mySwapCuda(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}
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

__device__ double mymin(const double a, const double b){
    if (a < b)
        return a;
    return b;
}


__global__ void back_ground_estimation(ImageView<rgb8> in, ImageView<rgb8> bg_value, ImageView<rgb8> candidate_value, int* time_matrix) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height) return;

    int idx = y * in.width + x;


}

__global__ void applyFlow(ImageView<rgb8> in, ImageView<rgb8> bg_value, ImageView<rgb8> candidate_value, int* time_matrix)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const double strictDistanceThreshold = 0.25;
    const double highlightDistanceMultiplier = 2.8; 

    if (x >= in.width || y >= in.height) return;

    rgb8 bg_pixel = bg_value.buffer[idx];
    rgb8 in_pixel = in.buffer[idx];
    rgb8 candidate_pixel = candidate_value.buffer[idx];

    Lab lab_in = rgbToLabGPU(in_pixel);
    Lab lab_bg = rgbToLabGPU(bg_pixel);

    double distance = deltaEGPU(lab_in, lab_bg);
    int time = time_matrix[idx];
    bool match = distance < 25.0;
    // int time = 0;
    if (match) {
        time = 0;
        bg_pixel.r = in_pixel.r;
        bg_pixel.g = in_pixel.g;
        bg_pixel.b = in_pixel.b;
        bg_value.buffer[idx] = bg_pixel;
    } else {
        if (time == 0) {
            candidate_pixel.r = in_pixel.r;
            candidate_pixel.g = in_pixel.g;
            candidate_pixel.b = in_pixel.b;
            candidate_value.buffer[idx] = candidate_pixel;
            time ++;
        } else if (time < 100) {
            candidate_pixel.r = (candidate_pixel.r + in_pixel.r) / 2;
            candidate_pixel.g = (candidate_pixel.g + in_pixel.g) / 2;
            candidate_pixel.b = (candidate_pixel.b + in_pixel.b) / 2;
            candidate_value.buffer[idx] = candidate_pixel;
            time++;
        } else {
            mySwapCuda(bg_pixel, candidate_pixel);
            bg_value.buffer[idx] = bg_pixel;
            candidate_value.buffer[idx] = candidate_pixel;
            time = 0;
        }
    }
    time_matrix[y * in.width + x] = time;

    // double distance = back_ground_estimation(in, bg_value, candidate_value, time_matrix);
    // int idx = y * in.width + x;
    // double distance = 0;
    if (distance < strictDistanceThreshold)
    {
        in.buffer[idx] = {0, 0, 0};
    }
    else
    {
        uint8_t intensity = static_cast<uint8_t>(mymin(255.0, distance * highlightDistanceMultiplier));
        in.buffer[idx] = {intensity, intensity, 0};
    }
    
}


void compute_cu(ImageView<rgb8> in) {
    cudaError_t err;
    static bool initialized = false;
    static Image<rgb8> device_bg;
    static Image<rgb8> device_candidate;
    int* time_matrix;

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
    // float* distances;
    // CUDA_CHECK(cudaMalloc(&distances, in.width * in.height * sizeof(float)));

    // Copy input image to device
    CUDA_CHECK(cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride,
                            in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice));

    if (!initialized) {
        // Initialize background and candidate images
        CUDA_CHECK(cudaMemcpy(device_bg.buffer, device_in.buffer, in.width * in.height * sizeof(rgb8), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(device_candidate.buffer, device_in.buffer, in.width * in.height * sizeof(rgb8), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMalloc(&time_matrix, in.width * in.height * sizeof(int)));
        initialized = true;
    }

    // Compute distances between background and input image
    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    
    

    std::cout << "before apply" << std::endl;
    applyFlow<<<grid, block>>>(device_in, device_bg, device_candidate, time_matrix);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erreur lors du lancement du filtre : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Synchroniser le dispositif
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erreur lors de la synchronisation du dispositif : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copier le résultat vers l'hôte
    err = cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Erreur lors de la copie de l'image traitée vers l'hôte : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        
    }
}

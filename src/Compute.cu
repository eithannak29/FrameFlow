#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"
#include <iostream>

__device__ double sRGBToLinearCUDA(double c) {
    return (c <= 0.04045) ? (c / 12.92) : pow((c + 0.055) / 1.055, 2.4);
}

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


__device__ void rgbToXyzCUDA(const rgb8& rgb, double& X, double& Y, double& Z) {
    double r = sRGBToLinearCUDA(rgb.r / 255.0);
    double g = sRGBToLinearCUDA(rgb.g / 255.0);
    double b = sRGBToLinearCUDA(rgb.b / 255.0);

    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

__device__ double f_xyz_to_labCUDA(double t) {
    const double epsilon = 0.008856;
    const double kappa = 903.3;
    return (t > epsilon) ? cbrt(t) : (kappa * t + 16.0) / 116.0;
}

__device__ Lab rgbToLabCUDA(const rgb8& rgb) {
    double X, Y, Z;
    rgbToXyzCUDA(rgb, X, Y, Z);

    const double Xr = 0.95047;
    const double Yr = 1.00000;
    const double Zr = 1.08883;

    double fx = f_xyz_to_labCUDA(X / Xr);
    double fy = f_xyz_to_labCUDA(Y / Yr);
    double fz = f_xyz_to_labCUDA(Z / Zr);

    Lab lab;
    lab.L = 116.0 * fy - 16.0;
    lab.a = 500.0 * (fx - fy);
    lab.b = 200.0 * (fy - fz);

    return lab;
}

__device__ double deltaECUDA(const Lab& lab1, const Lab& lab2) {
    double dL = lab1.L - lab2.L;
    double da = lab1.a - lab2.a;
    double db = lab1.b - lab2.b;
    return sqrt(dL * dL + da * da + db * db);
}

__device__ double mymin(const double a, const double b){
    if (a < b)
        return a;
    return b;
}


__device__ double back_ground_estimation(ImageView<rgb8> in, ImageView<uint8_t> bg_value, ImageView<uint8_t> candidate, int* time_matrix) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.width && y < in.height) {
        rgb8* in_pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride) + x;
        rgb8* bg_pixel = (rgb8*)((std::byte*)bg_value.buffer + y * bg_value.stride) + x;
        rgb8* candidate_pixel = (rgb8*)((std::byte*)candidate.buffer + y * candidate.stride) + x;

        Lab lab_in = rgbToLabCUDA(*in_pixel);
        Lab lab_bg = rgbToLabCUDA(*bg_pixel);

        double distance = deltaECUDA(lab_in, lab_bg);
        int time = time_matrix[y * in.width + x];
        bool match = distance < 25.0;

        if (match) {
            time = 0;
            *bg_pixel = *in_pixel;
        } else {
            if (time == 0) {
                candidate_pixel[x].r = in_pixel[x].r;
                candidate_pixel[x].g = in_pixel[x].g;
                candidate_pixel[x].b = in_pixel[x].b;
                time++;
            } else if (time < 100) {
                candidate_pixel[x].r = (candidate_pixel[x].r + in_pixel[x].r) / 2;
                candidate_pixel[x].g = (candidate_pixel[x].g + in_pixel[x].g) / 2;
                candidate_pixel[x].b = (candidate_pixel[x].b + in_pixel[x].b) / 2;
                time++;
            } else {
                mySwapCuda(*bg_pixel, *candidate_pixel);
                time = 0;
            }
        }
        time_matrix[y * in.width + x] = time;
    }
}

__global__ void applyFlow(ImageView<rgb8> in, ImageView<uint8_t> bg_value, ImageView<uint8_t> candidate, int* time_matrix)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const double strictDistanceThreshold = 0.25;
    const double highlightDistanceMultiplier = 2.8; 

    if (x < in.width && y < in.height)
    {
        double distance = back_ground_estimation(in, bg_value, candidate, time_matrix);
        rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        pixel[x].b = 0; 
        if (distance < strictDistanceThreshold)
        {
            pixel[x].r = 0; 
            pixel[x].g = 0;
        }
        else
        {
            uint8_t intensity = static_cast<uint8_t>(mymin(255.0, distance * highlightDistanceMultiplier));
            pixel[x].r = intensity;
            pixel[x].g= intensity;
        }
    }
}

void compute_cu(ImageView<rgb8> in )
{
    static Image<uint8_t> bg_value;
    static Image<uint8_t> candidate;
    int* time_matrix;
    std::cout << "begininng" << std::endl;
    if (bg_value.buffer == nullptr)
    {
        std::cout << "init" << std::endl;
        bg_value = Image<uint8_t>(in.width, in.height, true);
        candidate = Image<uint8_t>(in.width, in.height, true);
        cudaMemcpy2D(bg_value.buffer, bg_value.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
        // if (rc != cudaSuccess)
        // {
        //     std::cerr << "Error copying bg to device: " << cudaGetErrorString(rc) << std::endl;
        //     return;
        // }
        cudaMemcpy2D(candidate.buffer, candidate.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
        // if (rc != cudaSuccess)
        // {
        //     std::cerr << "Error copying candidate to device: " << cudaGetErrorString(rc) << std::endl;
        //     return;
        // }
        cudaMalloc(&time_matrix, in.width * in.height * sizeof(int));
        // if (rc != cudaSuccess)
        // {
        //     std::cerr << "Error allocating time_matrix on device: " << cudaGetErrorString(rc) << std::endl;
        //     return;
        // }
    }
    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    
    std::cout << "before device in" << std::endl;
    Image<rgb8> device_in(in.width, in.height, true);
    cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
    // if (rc != cudaSuccess)
    // {
    //     std::cerr << "Error copying in to device: " << cudaGetErrorString(rc) << std::endl;
    //     return;
    // }
    std::cout << "before apply" << std::endl;
    applyFlow<<<grid, block>>>(device_in, bg_value, candidate, time_matrix);
    std::cout << "before device to host" << std::endl;
    cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
    std::cout << "after device to host" << std::endl;
}

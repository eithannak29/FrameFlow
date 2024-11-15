#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"
#include <iostream>
#include <cmath> // For sqrtf and std::pow

//--------Template Functions--------

struct Lab
{
    double L, a, b;
};

template <typename T>
__device__ void mySwapCuda(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

template <typename T>
__device__ T myMinCuda(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
__device__ T myMaxCuda(T a, T b) {
    return a > b ? a : b;
}

__device__ double sRGBToLinear_cuda(double c) {
    if (c <= 0.04045)
        return c / 12.92;
    else
        return pow((c + 0.055) / 1.055, 2.4);
}

//------RGB/Lab conversion functions------

// Function for XYZ to Lab conversion
__device__ double f_xyz_to_lab_cuda(double t) {
    const double epsilon = 0.008856;
    const double kappa = 903.3;

    if (t > epsilon)
        return cbrt(t);
    else
        return (kappa * t + 16.0) / 116.0;
}

// Function to convert RGB to XYZ
__device__ void rgbToXyz_cuda(const rgb8& rgb, double& X, double& Y, double& Z) {
    double r = sRGBToLinear_cuda(rgb.r / 255.0);
    double g = sRGBToLinear_cuda(rgb.g / 255.0);
    double b = sRGBToLinear_cuda(rgb.b / 255.0);

    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

// Function to convert XYZ to Lab
__device__ Lab xyzToLab_cuda(double X, double Y, double Z) {
    const double Xr = 0.95047;
    const double Yr = 1.00000;
    const double Zr = 1.08883;

    double x = X / Xr;
    double y = Y / Yr;
    double z = Z / Zr;

    double fx = f_xyz_to_lab_cuda(x);
    double fy = f_xyz_to_lab_cuda(y);
    double fz = f_xyz_to_lab_cuda(z);

    Lab lab;
    lab.L = 116.0 * fy - 16.0;
    lab.a = 500.0 * (fx - fy);
    lab.b = 200.0 * (fy - fz);

    return lab;
}

// Function to convert RGB to Lab
__device__ Lab rgbToLab_cuda(const rgb8& rgb) {
    double X, Y, Z;
    rgbToXyz_cuda(rgb, X, Y, Z);
    return xyzToLab_cuda(X, Y, Z);
}

// Function to compute ΔE (CIE76) between two Lab colors
__device__ double deltaE_cuda(const Lab& lab1, const Lab& lab2) {
    double dL = lab1.L - lab2.L;
    double da = lab1.a - lab2.a;
    double db = lab1.b - lab2.b;
    return sqrt(dL * dL + da * da + db * db);
}

//--------CUDA Kernel Functions--------

__device__ double background_estimation(ImageView<rgb8> in, ImageView<rgb8> device_background, ImageView<rgb8> device_candidate, ImageView<uint8_t> pixel_time_counter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    rgb8* bg_pixel = (rgb8*)((std::byte*)device_background.buffer + y * device_background.stride);
    rgb8* candidate_pixel = (rgb8*)((std::byte*)device_candidate.buffer + y * device_candidate.stride);

    int sumR = 0, sumG = 0, sumB = 0, count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < in.width && ny >= 0 && ny < in.height) {
                rgb8 neighbor_pixel = *((rgb8*)((std::byte*)in.buffer + ny * in.stride) + nx);
                sumR += neighbor_pixel.r;
                sumG += neighbor_pixel.g;
                sumB += neighbor_pixel.b;
                count++;
            }
        }
    }
    rgb8 mean_pixel = {static_cast<uint8_t>(sumR / count),
                       static_cast<uint8_t>(sumG / count),
                       static_cast<uint8_t>(sumB / count)};

    double distance = deltaE_cuda(rgbToLab_cuda(pixel[x]), rgbToLab_cuda(bg_pixel[x]));
    bool match = distance < 2;

    uint8_t *time = (uint8_t*)((std::byte*)pixel_time_counter.buffer + y * pixel_time_counter.stride);

    if (!match) {
        if (time[x] == 0) {
            candidate_pixel[x] = pixel[x];
            time[x] += 1;
        } else if (time[x] < 100) {
            candidate_pixel[x].r = (candidate_pixel[x].r + mean_pixel.r) / 2;
            candidate_pixel[x].g = (candidate_pixel[x].g + mean_pixel.g) / 2;
            candidate_pixel[x].b = (candidate_pixel[x].b + mean_pixel.b) / 2;
            time[x] += 1;
        } else {
            mySwapCuda(bg_pixel[x].r, mean_pixel.r);
            mySwapCuda(bg_pixel[x].g, mean_pixel.g);
            mySwapCuda(bg_pixel[x].b, mean_pixel.b);
            time[x] = 0;
        }
    } else {
        bg_pixel[x].r = static_cast<uint8_t>(bg_pixel[x].r * 0.9 + mean_pixel.r * 0.1);
        bg_pixel[x].g = static_cast<uint8_t>(bg_pixel[x].g * 0.9 + mean_pixel.g * 0.1);
        bg_pixel[x].b = static_cast<uint8_t>(bg_pixel[x].b * 0.9 + mean_pixel.b * 0.1);
        time[x] = 0;
    }

    return distance;
}
__device__ void morphological(
    ImageView<rgb8> in,
    ImageView<rgb8> copy,
    const int* kernel,
    int radius,
    int diameter,
    bool erode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= radius && x < (in.width - radius) && y >= radius && y < (in.height - radius)) {
        rgb8* pixel = (rgb8*)((std::byte*)copy.buffer + y * copy.stride);

        uint8_t new_value = (erode) ? 255 : 0;
        for (int ky = 0; ky < diameter; ++ky) {
            for (int kx = 0; kx < diameter; ++kx) {
                if (kernel[ky * diameter + kx] == 1) {
                    int ny = y + ky - radius;
                    int nx = x + kx - radius;

                    rgb8* kernel_pixel = (rgb8*)((std::byte*)in.buffer + ny * in.stride);
                    if (erode) {
                        new_value = myMinCuda(new_value, static_cast<uint8_t>(kernel_pixel[nx].r * 0.8));
                    }
                    else {
                        new_value = myMaxCuda(new_value, static_cast<uint8_t>(kernel_pixel[nx].r * 0.8));
                    }
                }
            }
        }
        pixel[x].r = new_value;
    }
}

__global__ void morphologicalOpening(
    ImageView<rgb8> in,
    ImageView<rgb8> copy,
    const int* diskKernel,
    int radius,
    int diameter,
    bool erode)
{
    morphological(in, copy, diskKernel, radius, diameter, erode);
}

__device__ void hysteresis_threshold_process(ImageView<rgb8> in, int lowThreshold, int highThreshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    int intensity = pixel[x].r;

    if (intensity >= highThreshold) {
        pixel[x] = {255, 255, 255};
    } else if (intensity < lowThreshold) {
        pixel[x] = {0, 0, 0};
    } else {
        pixel[x] = {127, 127, 127};
    }
}

__device__ void propagate_edges(ImageView<rgb8> in, int lowThreshold, int highThreshold, bool* hasChanged) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

    if (pixel[x].r == 255) {
        int nb_neighbors = 0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;

                int neighborX = x + dx;
                int neighborY = y + dy;

                if (neighborX >= 0 && neighborX < in.width && neighborY >= 0 && neighborY < in.height) {
                    rgb8* neighborPixel = (rgb8*)((std::byte*)in.buffer + neighborY * in.stride);
                    int neighborIntensity = neighborPixel[neighborX].r;
                    if (neighborIntensity > lowThreshold && neighborIntensity < highThreshold && neighborPixel[neighborX].r == 127) {
                        neighborPixel[neighborX] = {255, 255, 255};
                        *hasChanged = true;
                    }
                    if (neighborIntensity >= lowThreshold) {
                        nb_neighbors++;
                    }
                }
            }
        }
        if (nb_neighbors < 4) {
            pixel[x] = {0, 0, 0};
            *hasChanged = true;
        }
    }
}

__global__ void hysteresis(ImageView<rgb8> in, int lowThreshold, int highThreshold) {
    hysteresis_threshold_process(in, lowThreshold, highThreshold);

    bool updated;
    do {
        updated = false;
        propagate_edges(in, lowThreshold, highThreshold, &updated);
    } while (updated);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

    if (pixel[x].r == 127) {
        pixel[x].r = 0;
        pixel[x].g = 0;
        pixel[x].b = 0;
    }
}

__global__ void background_estimation_process(
    ImageView<rgb8> in,
    ImageView<rgb8> device_background,
    ImageView<rgb8> device_candidate,
    ImageView<uint8_t> pixel_time_counter
    ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    double distance = background_estimation(in, device_background, device_candidate, pixel_time_counter);
    
    const double distanceMultiplier = 1.8;
    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    pixel[x].r = static_cast<uint8_t>(myMinCuda(255.0, distance * distanceMultiplier));
    
    rgb8* pixel_copy = (rgb8*)((std::byte*)copy.buffer + y * copy.stride);
    pixel_copy[x].r = static_cast<uint8_t>(myMinCuda(255.0, distance * distanceMultiplier));
}

__global__ void applyRedMask_cuda(ImageView<rgb8> in, ImageView<rgb8> initialPixels){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    rgb8* initial_pixel = (rgb8*)((std::byte*)initialPixels.buffer + y * initialPixels.stride);

    if (pixel[x].r > 0) {
        pixel[x].r = myMinCuda(255, static_cast<int>(initial_pixel[x].r + 0.5 * 255));
        pixel[x].g = initial_pixel[x].g;
        pixel[x].b = initial_pixel[x].b;
      }
    else {
        pixel[x].r = initial_pixel[x].r;
        pixel[x].g = initial_pixel[x].g;
        pixel[x].b = initial_pixel[x].b;
      }
}

void compute_cu(ImageView<rgb8> in)
{
    static Image<uint8_t> device_logo;
    static Image<uint8_t> pixel_time_counter;
    static Image<rgb8> device_background;
    static Image<rgb8> device_candidate;

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

    if (device_logo.buffer == nullptr)
    {
        device_logo = Image<uint8_t>(logo_width, logo_height, true);
        cudaMemcpy2D(device_logo.buffer, device_logo.stride, logo_data, logo_width, logo_width, logo_height, cudaMemcpyHostToDevice);
    }

    if (device_background.buffer == nullptr)
    {
        device_background = Image<rgb8>(in.width, in.height, true);
        cudaMemcpy2D(device_background.buffer, device_background.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);

        device_candidate = Image<rgb8>(in.width, in.height, true);
        cudaMemcpy2D(device_candidate.buffer, device_candidate.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);

        pixel_time_counter = Image<uint8_t>(in.width, in.height, true);
        cudaMemset2D(pixel_time_counter.buffer, pixel_time_counter.stride, 0, in.width, in.height);
    }

    Image<rgb8> device_in(in.width, in.height, true);
    cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);

    Image<rgb8> Initialcopy(in.width, in.height, true);
    cudaMemcpy2D(Initialcopy.buffer, Initialcopy.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);

    Image<rgb8> copy(in.width, in.height, true);
    cudaMemcpy2D(copy.buffer, copy.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToDevice);

    int min_dimension = std::min(in.width, in.height);
    int ratio_disk = 2;
    int radius = (min_dimension / 100) / ratio_disk;

    int diameter = 2 * radius + 1;
    int kernel_size = diameter * diameter;
    int* h_diskKernel = new int[kernel_size];

    int center = radius;
    for (int i = 0; i < diameter; ++i) {
        for (int j = 0; j < diameter; ++j) {
            if (sqrtf((i - center) * (i - center) + (j - center) * (j - center)) <= radius) {
                h_diskKernel[i * diameter + j] = 1;
            }
            else {
                h_diskKernel[i * diameter + j] = 0;
            }
        }
    }
l
    int* d_diskKernel;
    cudaMalloc(&d_diskKernel, kernel_size * sizeof(int));
    cudaMemcpy(d_diskKernel, h_diskKernel, kernel_size * sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_diskKernel;

    background_estimation_process<<<grid, block>>>(
        device_in,
        device_background,
        device_candidate,
        pixel_time_counter,
        copy,
        d_diskKernel,
        radius,
        diameter);

    cudaDeviceSynchronize();

    morphologicalOpening<<<grid, block>>>(device_in, copy, d_diskKernel, radius, diameter, true);
    cudaDeviceSynchronize();

    morphologicalOpening<<<grid, block>>>(device_in, copy, d_diskKernel, radius, diameter, false);
    cudaDeviceSynchronize();


    hysteresis<<<grid, block>>>(device_in, 15, 30);
    cudaDeviceSynchronize();

    applyRedMask_cuda<<<grid, block>>>(device_in, Initialcopy);
    cudaDeviceSynchronize();

    cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);

    cudaFree(d_diskKernel);
}

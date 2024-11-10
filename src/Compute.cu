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


__device__ double back_ground_estimation(ImageView<rgb8> in, ImageView<rgb8> bg_value, ImageView<rgb8> candidate, int* time_matrix) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * in.width + x;

    rgb8 bg_pixel = bg_value.buffer[idx];
    rgb8 in_pixel = in.buffer[idx];
    rgb8 candidate_pixel = candidate_value.buffer[idx];

    Lab lab_in = rgbToLabCUDA(in_pixel);
    Lab lab_bg = rgbToLabCUDA(bg_pixel);

    double distance = deltaECUDA(lab_in, lab_bg);
    int time = time_matrix[y * in.width + x];
    bool match = distance < 25.0;

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
            time++;
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
}

__global__ void applyFlow(ImageView<rgb8> in, ImageView<rgb8> bg_value, ImageView<rgb8> candidate, int* time_matrix)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const double strictDistanceThreshold = 0.25;
    const double highlightDistanceMultiplier = 2.8; 

    if (x >= width || y >= height) return;

    double distance = back_ground_estimation(in, bg_value, candidate, time_matrix);
    
    int idx = y * in.width + x;
    if (distance < strictDistanceThreshold)
    {
        in_buffer[idx] = {0, 0, 0};
    }
    else
    {
        uint8_t intensity = static_cast<uint8_t>(mymin(255.0, distance * highlightDistanceMultiplier));
        in_buffer[idx] = {intensity, intensity, 0};
    }
    
}

void compute_cu(ImageView<rgb8> in )
{
    cudaError_t err;
    static Image<rgb8> bg_value;
    static Image<rgb8> candidate;
    int* time_matrix;
    std::cout << "begininng" << std::endl;
    if (bg_value.buffer == nullptr)
    {
        std::cout << "init" << std::endl;
        bg_value = Image<rgb8>(in.width, in.height, true);
        candidate = Image<rgb8>(in.width, in.height, true);
        err = cudaMemcpy2D(bg_value.buffer, bg_value.stride, in.buffer, in.width, in.width, in.height, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Erreur d'allocation de bg_value : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpy2D(candidate.buffer, candidate.stride, in.buffer, in.width, in.width , in.height, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Erreur d'allocation de canditate : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMalloc(&time_matrix, in.width * in.height * sizeof(int));
        if (err != cudaSuccess) {
            fprintf(stderr, "Erreur d'allocation de time matrix : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    
    std::cout << "before device in" << std::endl;
    Image<rgb8> device_in(in.width, in.height, true);
    err = cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
            fprintf(stderr, "Erreur d'allocation de device_in: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    std::cout << "before apply" << std::endl;
    applyFlow<<<grid, block>>>(device_in, bg_value, candidate, time_matrix);
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

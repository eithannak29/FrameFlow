#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"


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




// void compute_cu(ImageView<rgb8> in)
// {
//     static Image<uint8_t> device_logo;
//     static

//     dim3 block(16, 16);
//     dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    
//     // Copy the logo to the device if it is not already there
//     if (device_logo.buffer == nullptr)
//     {
//         device_logo = Image<uint8_t>(logo_width, logo_height, true);
//         cudaMemcpy2D(device_logo.buffer, device_logo.stride, logo_data, logo_width, logo_width, logo_height, cudaMemcpyHostToDevice);
//     }

//     // Copy the input image to the device
//     Image<rgb8> device_in(in.width, in.height, true);
//     cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
    
//     mykernel<<<grid, block>>>(device_in, device_logo);

//     // Copy the result back to the host
//     cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
// }
__device__ void back_ground_estimation(ImageView<rgb8> in, ImageView<rgb8> bg_value, ImageView<rgb8> candidate, int* time_matrix) {
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
                candidate_pixel->r = in_pixel->r;
                candidate_pixel->g = in_pixel->g;
                candidate_pixel->b = in_pixel->b;
                time++;
            } else if (time < 100) {
                candidate_pixel->r = (candidate_pixel->r + in_pixel->r) / 2;
                candidate_pixel->g = (candidate_pixel->g + in_pixel->g) / 2;
                candidate_pixel->b = (candidate_pixel->b + in_pixel->b) / 2;
                time++;
            } else {
                mySwapCuda(*bg_pixel, *candidate_pixel);
                time = 0;
            }
        }
        time_matrix[y * in.width + x] = time;
    }
}

__global__ void applyFlow(ImageView<rgb8> in, ImageView<rgb8> bg_value, ImageView<rgb8> candidate, int* time_matrix)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const double strictDistanceThreshold = 0.25;
    const double highlightDistanceMultiplier = 2.8; 

    if (x < in.width && y < in.height)
    {
        double distance = back_ground_estimation(in, bg_value, candidate, time_matrix);
        if (distance < strictDistanceThreshold)
        {
            in.buffer[y * in.stride + x] = {0, 0, 0};
        }
        else
        {
            uint8_t intensity = static_cast<uint8_t>(std::min(255.0, distance * highlightDistanceMultiplier));
            in.buffer[y * in.stride + x] = {intensity, intensity, 0};
        }
    }
}

void compute_cu(ImageView<rgb8> in )
{
    static ImageView<rgb8> bg_value;
    static ImageView<rgb8> candidate;
    int* time_matrix;
    if (bg_value.buffer == nullptr)
    {
        bg_value = ImageView<rgb8>(in.width, in.height, true);
        candidate = ImageView<rgb8>(in.width, in.height, true);
        rc = cudaMemcpy2D(bg_value.buffer, bg_value.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
        // if (rc != cudaSuccess)
        // {
        //     std::cerr << "Error copying bg to device: " << cudaGetErrorString(rc) << std::endl;
        //     return;
        // }
        rc = cudaMemcpy2D(candidate.buffer, candidate.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
        // if (rc != cudaSuccess)
        // {
        //     std::cerr << "Error copying candidate to device: " << cudaGetErrorString(rc) << std::endl;
        //     return;
        // }
        rc = cudaMalloc(&time_matrix, in.width * in.height * sizeof(int));
        // if (rc != cudaSuccess)
        // {
        //     std::cerr << "Error allocating time_matrix on device: " << cudaGetErrorString(rc) << std::endl;
        //     return;
        // }
    }
    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

    Image<rgb8> device_in(in.width, in.height, true);
    rc = cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
    // if (rc != cudaSuccess)
    // {
    //     std::cerr << "Error copying in to device: " << cudaGetErrorString(rc) << std::endl;
    //     return;
    // }
    applyFlow<<<grid, block>>>(device_in, bg_value, candidate, time_matrix);
    cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
}
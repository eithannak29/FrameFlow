#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"
#include <iostream>


struct Lab
{
    double L, a, b;
};

// Single threaded version of the Method
__global__ void mykernel(ImageView<rgb8> in, ImageView<uint8_t> logo)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < in.width && y < in.height)
    {
        rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        pixel[x].r = 0;

        if (x < logo.width && y < logo.height)
        {
            float alpha = logo.buffer[y * logo.stride + x] / 255.f;
            pixel[x].g = uint8_t(alpha * pixel[x].g + (1 - alpha) * 255);
            pixel[x].b = uint8_t(alpha * pixel[x].b + (1 - alpha) * 255);
        }
    }
}


__device__ double sRGBToLinear_cuda(double c) {
    if (c <= 0.04045)
        return c / 12.92;
    else
        return std::pow((c + 0.055) / 1.055, 2.4);
}

// Fonction auxiliaire pour la conversion XYZ -> Lab
__device__ double f_xyz_to_lab_cuda(double t) {
    const double epsilon = 0.008856; // (6/29)^3
    const double kappa = 903.3;      // (29/3)^3

    if (t > epsilon)
        return std::cbrt(t); // Racine cubique
    else
        return (kappa * t + 16.0) / 116.0;
}

// Fonction pour convertir RGB en XYZ
__device__ void rgbToXyz_cuda(const rgb8& rgb, double& X, double& Y, double& Z) {
    // Normalisation des valeurs RGB entre 0 et 1
    double r = sRGBToLinear_cuda(rgb.r / 255.0);
    double g = sRGBToLinear_cuda(rgb.g / 255.0);
    double b = sRGBToLinear_cuda(rgb.b / 255.0);

    // Matrice de conversion sRGB D65
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}
// Fonction pour convertir XYZ en Lab
__device__ Lab xyzToLab_cuda(double X, double Y, double Z) {
    // Blanc de référence D65
    const double Xr = 0.95047;
    const double Yr = 1.00000;
    const double Zr = 1.08883;

    // Normalisation par rapport au blanc de référence
    double x = X / Xr;
    double y = Y / Yr;
    double z = Z / Zr;

    // Application de la fonction f(t)
    double fx = f_xyz_to_lab_cuda(x);
    double fy = f_xyz_to_lab_cuda(y);
    double fz = f_xyz_to_lab_cuda(z);

    Lab lab;
    lab.L = 116.0 * fy - 16.0;
    lab.a = 500.0 * (fx - fy);
    lab.b = 200.0 * (fy - fz);

    return lab;
}

// Fonction pour convertir RGB en Lab
__device__ Lab rgbToLab_cuda(const rgb8& rgb) {
    double X, Y, Z;
    rgbToXyz_cuda(rgb, X, Y, Z);
    return xyzToLab_cuda(X, Y, Z);
}

// Fonction pour calculer la distance ΔE (CIE76) entre deux couleurs Lab
__device__ double deltaE_cuda(const Lab& lab1, const Lab& lab2) {
    double dL = lab1.L - lab2.L;
    double da = lab1.a - lab2.a;
    double db = lab1.b - lab2.b;
    return std::sqrt(dL * dL + da * da + db * db);
}

__device__ double background_estimation(ImageView<rgb8> in)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    rgb8* bg_pixel = (rgb8*)((std::byte*)device_background.buffer + y * device_background.stride);
    rgb8* candidate_pixel = (rgb8*)((std::byte*)device_candidate.buffer + y * device_candidate.stride);

    double distance = deltaE(rgbToLab(pixel[x]), rgbToLab(bg_pixel[x]));
    bool match = distance < 25;

    uint8_t *time = (uint8_t*)((std::byte*)pixel_time_counter.buffer + y * pixel_time_counter.stride);

    if (!match)
    {
        if(time[x] == 0)
        {
            candidate_pixel[x].r = pixel[x].r;
            candidate_pixel[x].g = pixel[x].g;
            candidate_pixel[x].b = pixel[x].b;
            time[x] += 1;
        }
        else if (time[x] < 100)
        {
            candidate_pixel[x].r = (candidate_pixel[x].r + pixel[x].r) / 2;
            candidate_pixel[x].g = (candidate_pixel[x].g + pixel[x].g) / 2;
            candidate_pixel[x].b = (candidate_pixel[x].b + pixel[x].b) / 2;
            time[x] += 1;
        }
        else
        {
            std::swap(bg_pixel[x].r, candidate_pixel[x].r);
            std::swap(bg_pixel[x].g, candidate_pixel[x].g);
            std::swap(bg_pixel[x].b, candidate_pixel[x].b);
            time[x] = 0;        
        }
    }
    else {
        bg_pixel[x].r = (bg_pixel[x].r + pixel[x].r) / 2;
        bg_pixel[x].g = (bg_pixel[x].g + pixel[x].g) / 2;
        bg_pixel[x].b = (bg_pixel[x].b + pixel[x].b) / 2; 
        time[x] = 0;
    }
    printf("Distance: %f\n", distance);
    return distance;
}

__global__ void background_estimation_process(ImageView<rgb8> in)
{
    const double treshold = 25;
    const double distanceMultiplier = 2.8;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    double distance = background_estimation(in);

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

    pixel[x].r = static_cast<uint8_t>(std::min(255.0, distance * distanceMultiplier));
}


void compute_cu(ImageView<rgb8> in)
{
    static Image<uint8_t> device_logo;
    static Image<uint8_t> pixel_time_counter;
    static Image<rgb8> device_background;
    static Image<rgb8> device_candidate;

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    
    // Copy the logo to the device if it is not already there
    if (device_logo.buffer == nullptr)
    {
        device_logo = Image<uint8_t>(logo_width, logo_height, true);
        cudaMemcpy2D(device_logo.buffer, device_logo.stride, logo_data, logo_width, logo_width, logo_height, cudaMemcpyHostToDevice);
    }

    if(device_background.buffer == nullptr)
    {
        device_background = Image<rgb8>(in.width, in.height, true);
        cudaMemcpy2D(device_background.buffer, device_background.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);

        device_candidate = Image<rgb8>(in.width, in.height, true);
        cudaMemcpy2D(device_candidate.buffer, device_candidate.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);

        pixel_time_counter = Image<uint8_t>(in.width, in.height, true);
        cudaMemset2D(pixel_time_counter.buffer, pixel_time_counter.stride, 0, in.width, in.height);
    }

    // Copy the input image to the device
    Image<rgb8> device_in(in.width, in.height, true);
    cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice);
    
    // mykernel<<<grid, block>>>(device_background, device_logo);

    background_estimation_process<<<grid, block>>>(device_in);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy2D(in.buffer, in.stride, device_background.buffer, device_background.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
}

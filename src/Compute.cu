#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"
#include <iostream>

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

__device__ double background_estimation(ImageView<rgb8> in, ImageView<rgb8> device_background, ImageView<rgb8> device_candidate, ImageView<uint8_t> pixel_time_counter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    rgb8* bg_pixel = (rgb8*)((std::byte*)device_background .buffer + y * device_background.stride);
    rgb8* candidate_pixel = (rgb8*)((std::byte*)device_candidate.buffer + y * device_candidate.stride);

    double distance = deltaE_cuda(rgbToLab_cuda(pixel[x]), rgbToLab_cuda(bg_pixel[x]));
    bool match = distance < 25;

    // std::cout << "aled: " << distance << std::endl;

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
        else if (time[x] < 25)
        {
            candidate_pixel[x].r = (candidate_pixel[x].r + pixel[x].r) / 2;
            candidate_pixel[x].g = (candidate_pixel[x].g + pixel[x].g) / 2;
            candidate_pixel[x].b = (candidate_pixel[x].b + pixel[x].b) / 2;
            time[x] += 1;
        }
        else
        {
            mySwapCuda(bg_pixel[x].r, candidate_pixel[x].r);
            mySwapCuda(bg_pixel[x].g, candidate_pixel[x].g);
            mySwapCuda(bg_pixel[x].b, candidate_pixel[x].b);
            time[x] = 0;        
        }
    }
    else {
        bg_pixel[x].r = (bg_pixel[x].r + pixel[x].r) / 2;
        bg_pixel[x].g = (bg_pixel[x].g + pixel[x].g) / 2;
        bg_pixel[x].b = (bg_pixel[x].b + pixel[x].b) / 2; 
        time[x] = 0;
    }
    return distance;
}

__device__ void apply_filter(ImageView<rgb8> in, double distance){

    const double distanceMultiplier = 2.8;
    const double threshold = 3.5;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

    // if (distance > threshold)
    // {
    //     uint8_t intensity = static_cast<uint8_t>(myMinCuda(distance * distanceMultiplier, 255.0));
    //     pixel[x].r = intensity;
    //     pixel[x].g = intensity;
    //     pixel[x].b = 0;
    // }
    // else
    // {
    //     pixel[x].r = 0;
    //     pixel[x].g = 0;
    //     pixel[x].b = 0;
    // }
             
    pixel[x].r = static_cast<uint8_t>(myMinCuda(255.0, distance * distanceMultiplier));

}


__device__ int* createDiskKernel_cuda(int radius) {
    int diameter = 2 * radius + 1;
    int* kernel = new int[diameter * diameter];
    int center = radius;
    
    for (int i = 0; i < diameter; ++i) {
        for (int j = 0; j < diameter; ++j) {
            if (sqrtf((i - center) * (i - center) + (j - center) * (j - center)) <= radius) {
                kernel[i * diameter + j] = 1;
            }
            else {
                kernel[i * diameter + j] = 0;
            }
        }
    }
    return kernel;
}

// __global__ void createKernelDevice(int radius, int* kernel) {
//     int diameter = 2 * radius + 1;
//     //int* kernel = new int[diameter * diameter];
//     int center = radius;
    
//     for (int i = 0; i < diameter; ++i) {
//         for (int j = 0; j < diameter; ++j) {
//             if (sqrtf((i - center) * (i - center) + (j - center) * (j - center)) <= radius) {
//                 kernel[i * diameter + j] = 1;
//             }
//             else {
//                 kernel[i * diameter + j] = 0;
//             }
//         }
//     }
// }

// __device__ Image<rgb8> clone(ImageView<rgb8> in)
// {
//     Image<rgb8> img = Image<rgb8>();
//     img.buffer = in.buffer;
//     img.width = in.width;
//     img.height = in.height;
//     img.stride = in.width;
//     //return img.clone();

// }

// Appliquer une opération d'érosion
__device__ void morphological(ImageView<rgb8> in, ImageView<rgb8> copy, const int* kernel, int radius, bool erode) {
    //Image<rgb8> copy = clone(in);  // Faire une copie temporaire de l'image pour éviter la corruption
    
    int diameter = 2 * radius + 1;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Vérifiez si (x, y) est dans les limites et respecte le rayon
    if (x >= radius && x < (in.width - radius) && y >= radius && y < (in.height - radius)) {
        rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        
        uint8_t new_value = (erode) ? 255 : 0;
        for (int ky = 0; ky < diameter; ++ky) {
            for (int kx = 0; kx < diameter; ++kx) {

                if (kernel[ky * diameter + kx] == 1) {
                    int ny = y + ky - radius;
                    int nx = x + kx - radius;

                    rgb8* kernel_pixel = (rgb8*)((std::byte*)in.buffer + ny * in.stride);
                    if (erode){
                        new_value = myMinCuda(new_value, kernel_pixel[nx].r);
                    }
                    else {
                        new_value = myMaxCuda(new_value, kernel_pixel[nx].r);   
                    }                 
                }
            }
        }
        pixel = (rgb8*)((std::byte*)copy.buffer + y * copy.stride);
        pixel[x].r = new_value;
    }
}

__device__ void morphologicalOpening(ImageView<rgb8> in, ImageView<rgb8> copy, int minradius) {
    int min_dimension = myMinCuda(in.width, in.height);
    int ratio_disk = 1; // 1 % de la resolution de l'image
    int radius = myMaxCuda(minradius, (min_dimension / 100) * ratio_disk);
    // std::cout << "radius: " << radius << std::endl;
    // Créer un noyau en forme de disque avec le rayon calculé
    auto diskKernel = createDiskKernel_cuda(radius);
    // Étape 1 : Erosion
    morphological(in, copy, diskKernel, radius, true);
    // Étape 2 : Dilatation
    morphological(in, copy, diskKernel, radius, false);
}

__global__ void background_estimation_process(ImageView<rgb8> in, ImageView<rgb8> device_background, ImageView<rgb8> device_candidate, ImageView<uint8_t> pixel_time_counter, ImageView<rgb8> copy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // dim3 block(16, 16);
    // dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

    if (x >= in.width || y >= in.height)
        return;

    double distance = background_estimation(in, device_background, device_candidate, pixel_time_counter);
    // if (distance > 1) {
    //     printf("distance: %f\n", distance);
    // }
    //rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

    //pixel[x].r = static_cast<uint8_t>(myMinCuda(distance * distanceMultiplier, 255.0));

    apply_filter(in, distance);


    // Étape 1 : Erosion
    

    //morphologicalOpening(in, copy, 3);

    //apply_filter<<<grid, block>>>(in, distance);
}


void compute_cu(ImageView<rgb8> in)
{
    static Image<uint8_t> device_logo;
    static Image<uint8_t> pixel_time_counter;
    static Image<rgb8> device_background;
    static Image<rgb8> device_candidate;

    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);

    // Create a copy of the input image
    //----------
    
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

    Image<rgb8> copy(in.width, in.height, true); 
    cudaMemcpy2D(copy.buffer, copy.stride, in.buffer, in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToDevice);

    background_estimation_process<<<grid, block>>>(device_in, device_background, device_candidate, pixel_time_counter, copy);

    cudaDeviceSynchronize();


    

    //cudaMemcpy(device_in.buffer, device_in.stride, copy.buffer, in.width * in.height * sizeof(rgb8), cudaMemcpyDeviceToDevice);

    // Copy the result back to the host
    //cudaMemcpy2D(in.buffer, in.stride, device_background.buffer, device_background.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost);
}
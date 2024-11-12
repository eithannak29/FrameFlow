#include "Image.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>

std::vector<std::vector<int>> createDiskKernel(int radius) {
    int diameter = 2 * radius + 1;
    std::vector<std::vector<int>> kernel(diameter, std::vector<int>(diameter, 0));
    int center = radius;
    
    for (int i = 0; i < diameter; ++i) {
        for (int j = 0; j < diameter; ++j) {
            if (std::sqrt((i - center) * (i - center) + (j - center) * (j - center)) <= radius) {
                kernel[i][j] = 1;
            }
        }
    }
    return kernel;
}

Image<rgb8> clone(ImageView<rgb8> in)
{
    Image<rgb8> img = Image<rgb8>();
    img.buffer = in.buffer;
    img.width = in.width;
    img.height = in.height;
    img.stride = in.width;
    return img.clone();
}

// Appliquer une opération d'érosion
void erode(ImageView<rgb8> in, const std::vector<std::vector<int>>& kernel, int radius) {
    Image<rgb8> copy = clone(in);  // Faire une copie temporaire de l'image pour éviter la corruption
    int diameter = 2 * radius + 1;
    for (int y = radius; y < in.height - radius; ++y) {
        rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        for (int x = radius; x < in.width - radius; ++x) {
            uint8_t min_pixel = 255;
            for (int ky = 0; ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {

                    if (kernel[ky][kx] == 1) {
                        int ny = y + ky - radius;
                        int nx = x + kx - radius;

                        rgb8* kernel_pixel = (rgb8*)((std::byte*)in.buffer + ny * in.stride);
                        min_pixel = std::min(min_pixel, kernel_pixel[nx].r);                      
                        std::cout << "val: " << (int)kernel_pixel[nx].r << std::endl;
                        }
                    }
                }
          pixel = (rgb8*)((std::byte*)copy.buffer + y * copy.stride);
          pixel[x].r = min_pixel;
          }
        }
    in = copy;  
    }

// Appliquer une opération de dilatation
void dilate(ImageView<rgb8> in, const std::vector<std::vector<int>>& kernel, int radius) {
    int diameter = 2 * radius + 1;
    Image<rgb8> copy = clone(in);
    for (int y = radius; y < in.height - radius; ++y) {
        for (int x = radius; x < in.width - radius; ++x) {
            uint8_t max_pixel = 0;
            for (int ky = 0; ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {
                    if (kernel[ky][kx] == 1){
                        int ny = y + ky - radius;
                        int nx = x + kx - radius;
                        std::cout << "max_pixel: " << max_pixel << std::endl;
                        rgb8 kernel_pixel = in.buffer[ny * in.width + nx];
                        max_pixel = std::max(max_pixel, kernel_pixel.r);
                        std::cout << "val: " << (int)kernel_pixel.r << std::endl;
                        std::cout << "max_pixel: " << max_pixel << std::endl;
                    }
                }
            }
            rgb8* pixel = (rgb8*)((std::byte*)copy.buffer + y * copy.stride);
            pixel[x].r = max_pixel;
            std::cout << "val: " << max_pixel << std::endl;
            
        }
    }
    in = copy;  // Appliquer la copie modifiée à l'image d'origine
}

// Ouverture morphologique (érosion suivie de dilatation)
void morphologicalOpening(ImageView<rgb8> in, int minradius) {
    int min_dimension = std::min(in.width, in.height);
    int ratio_disk = 1; // 1 % de la resolution de l'image
    int radius = std::max(minradius, (min_dimension / 100) * ratio_disk);
    // std::cout << "radius: " << radius << std::endl;
    // Créer un noyau en forme de disque avec le rayon calculé
    auto diskKernel = createDiskKernel(radius);
    // Étape 1 : Erosion
    // erode(in, diskKernel, radius);
    // Étape 2 : Dilatation
    dilate(in, diskKernel, radius);
}

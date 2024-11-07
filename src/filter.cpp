#include "Image.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>

// Appliquer une opération d'érosion
void erode(ImageView<rgb8>& image, int radius) {
    ImageView<rgb8> copy = image;  // Faire une copie temporaire de l'image pour éviter la corruption
    int diameter = 2 * radius + 1;

    for (int y = radius; y < image.height - radius; ++y) {
        for (int x = radius; x < image.width - radius; ++x) {
            bool erosion = false;
            rgb8& pixel = image.buffer[y * image.stride + x];
            if(pixel.r == 0){
                continue;
            }
            for (int ky = 0; !erosion && ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {
                    int ny = y + ky - radius;
                    int nx = x + kx - radius;
                    rgb8 kernel_pixel = image.buffer[ny * image.stride + nx];
                    if (kernel_pixel.r == 0) {
                        erosion = true;                        
                    }
                }
            }
            if (erosion) {
                rgb8& pixel = copy.buffer[y * copy.stride + x];
                pixel.r = 0;
                pixel.b = pixel.g;
            }
        }
        image = copy;  // Appliquer la copie modifiée à l'image d'origine
    }
}

// Appliquer une opération de dilatation
void dilate(ImageView<rgb8>& image, int radius) {
    ImageView<rgb8> copy = image;  // Faire une copie temporaire de l'image pour éviter la corruption
    int diameter = 2 * radius + 1;

    for (int y = radius; y < image.height - radius; ++y) {
        for (int x = radius; x < image.width - radius; ++x) {
            bool dilatation = false;
            rgb8& pixel = image.buffer[y * image.stride + x];
            if(pixel.b == 0){
                continue;
            }
            for (int ky = 0; !dilatation && ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {
                    int ny = y + ky - radius;
                    int nx = x + kx - radius;
                    rgb8 kernel_pixel = image.buffer[ny * image.stride + nx];
                    if (kernel_pixel.b == 0) {
                        dilatation = true;                        
                    }
                }
            }
            if (dilatation) {
                rgb8& pixel = copy.buffer[y * copy.stride + x];
                pixel.r = pixel.g;
                pixel.b = 0;
            }
        }
    }
    
    image = copy;  // Appliquer la copie modifiée à l'image d'origine
}

// Ouverture morphologique (érosion suivie de dilatation)
void morphologicalOpening(ImageView<rgb8>& image, int radius) {
    // std::cout << "start morphologie"  << std::endl;
    // Étape 1 : Erosion
    erode(image,  radius);

    // Étape 2 : Dilatation
    dilate(image, radius);
}
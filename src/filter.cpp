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


// Appliquer une opération d'érosion
void erode(ImageView<rgb8>& image, const std::vector<std::vector<int>>& kernel, int radius) {
    ImageView<rgb8> copy = image;  // Faire une copie temporaire de l'image pour éviter la corruption
    int diameter = 2 * radius + 1;
    for (int y = radius; y < image.height - radius; ++y) {
        for (int x = radius; x < image.width - radius; ++x) {
            rgb8& pixel = image.buffer[y * image.width + x];
            if(pixel.r == 0){
                continue;
            }
            uint8_t min_pixel = 255;
            for (int ky = 0; ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {

                    if (kernel[ky][kx] == 1) {
                        int ny = y + ky - radius;
                        int nx = x + kx - radius;
                    
                        rgb8 kernel_pixel = image.buffer[ny * image.width + nx];
                        min_pixel = std::min(min_pixel, kernel_pixel.r);                      
                        }
                    }
                }
            pixel = copy.buffer[y * copy.width + x];
            pixel.r = min_pixel;
            }
        }
    }

// Appliquer une opération de dilatation
void dilate(ImageView<rgb8>& in, const std::vector<std::vector<int>>& kernel, int radius) {
    int diameter = 2 * radius + 1;
    ImageView<rgb8> copy = in;
    for (int y = radius; y < in.height - radius; ++y) {
        for (int x = radius; x < in.width - radius; ++x) {
            rgb8& pixel = copy.buffer[y * in.width + x];
            if(pixel.r != 0){
                continue;
            }
            uint8_t max_pixel = 0;
            for (int ky = 0; ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {
                    if (kernel[ky][kx] == 1){
                        int ny = y + ky - radius;
                        int nx = x + kx - radius;
                        rgb8 kernel_pixel = in.buffer[ny * in.width + nx];
                        max_pixel = std::max(max_pixel, kernel_pixel.r);
                    }
                }
            }
            pixel = copy.buffer[y * copy.width + x];
            pixel.r = max_pixel;
        }
    }
    
    in = copy;  // Appliquer la copie modifiée à l'image d'origine
}

// Ouverture morphologique (érosion suivie de dilatation)
void morphologicalOpening(ImageView<rgb8>& in, int minradius) {
    int min_dimension = std::min(in.width, in.height);
    int ratio_disk = 1; // 1 % de la resolution de l'image
    int radius = std::max(minradius, (min_dimension / 100) * ratio_disk); 

    // Créer un noyau en forme de disque avec le rayon calculé
    auto diskKernel = createDiskKernel(radius);
    // Étape 1 : Erosion
    erode(in, diskKernel, radius);
    // Étape 2 : Dilatation
    dilate(in, diskKernel, radius);
}

ImageView<rgb8> HysteresisThreshold(ImageView<rgb8> in, int lowThreshold, int highThreshold) {

  // Créer une queue pour propager les pixels de bord fort
  std::queue<std::pair<int, int>> edgeQueue;

  // Initialisation : marquer les bords forts et ajouter les pixels forts dans la queue
  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;
      rgb8 pixel = in.buffer[index];
      int intensity = (pixel.r + pixel.g + pixel.b) / 3;

      if (intensity >= highThreshold) {
        // Marque le pixel comme bord fort (blanc) et l'ajoute dans la queue
        in.buffer[index] = {255, 255, 255};
        edgeQueue.push({x, y});
      } else if (intensity < lowThreshold) {
        // Marque les pixels en dessous du seuil bas comme arrière-plan (noir)
        in.buffer[index] = {0, 0, 0};
      } else {
        // Les pixels entre les seuils sont des candidats potentiels (gris)
        in.buffer[index] = {127, 127, 127};
      }
    }
  }

  // Propagation des bords forts vers les pixels adjacents en fonction du seuil bas
  while (!edgeQueue.empty()) {
    auto [x, y] = edgeQueue.front();
    edgeQueue.pop();

    // Parcours des pixels voisins
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;

        int neighborX = x + dx;
        int neighborY = y + dy;

        if (neighborX >= 0 && neighborX < in.width && neighborY >= 0 && neighborY < in.height) {
          int neighborIndex = neighborY * in.width + neighborX;
          rgb8& neighborPixel = in.buffer[neighborIndex];

          int neighborIntensity = (neighborPixel.r + neighborPixel.g + neighborPixel.b) / 3;

          // Si le pixel voisin est entre les seuils et qu'il n'est pas déjà marqué comme bord fort
          if (neighborIntensity >= lowThreshold && neighborIntensity < highThreshold && neighborPixel.r != 255) {
            // Marque le pixel comme bord fort et l'ajoute dans la queue pour continuer la propagation
            neighborPixel = {255, 255, 255};
            edgeQueue.push({neighborX, neighborY});
          }
        }
      }
    }
  }

  // Finalise l'image : tous les pixels gris (127, 127, 127) restants deviennent arrière-plan (noir)
  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;
      rgb8& pixel = in.buffer[index];

      if (pixel.r == 127 && pixel.g == 127 && pixel.b == 127) {
        pixel = {0, 0, 0};
      }
    }
  }

  return in;
}

ImageView<rgb8> applyRedMask(ImageView<rgb8> in, const ImageView<rgb8>& mask, std::vector<rgb8> initialPixels) {
  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;

      if (mask.buffer[index].r > 0) {
        in.buffer[index].r = std::min(255, static_cast<int>(initialPixels[index].r + 0.5 * 255));
        in.buffer[index].g = initialPixels[index].g;
        in.buffer[index].b = initialPixels[index].b;
      }
      else {
        in.buffer[index] = initialPixels[index];
      }
    }
  }

  return in;
}

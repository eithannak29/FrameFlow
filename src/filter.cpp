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
    // std::cout << image.stride << "height " << image.height << "width" << image.width << std::endl;
    for (int y = radius; y < image.height - radius; ++y) {
        for (int x = radius; x < image.width - radius; ++x) {
            bool erosion = false;
            // std::cout << "before pixel" << std::endl;
            // std::cout << "before pixel" << std::endl;
            //if (y   <0 || y  >= image.height || x < 0 || x >= image.width){
            // if (y * image.stride + x >= image.height * image.width){
            // std::cout << "Index out of bounds : (y: " << y << ", x: " << x << ")" << std::endl;
            //}
            rgb8& pixel = image.buffer[y * image.width + x];
            
            //std::cout << "x: " << x << " y: " << y << " aled:" << (image.width * y + x) << std::endl;
            if(pixel.r == 0){
    
                // std::cout << "pixel is activated" << std::endl;
                continue;
            }
            // std::cout << "pixel is activated" << std::endl;
            for (int ky = 0; !erosion && ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {

                    // std::cout << "pixel kernel" << std::endl;
                    if (kernel[ky][kx] == 1) {
                        int ny = y + ky - radius;
                        int nx = x + kx - radius;
                    
                        rgb8 kernel_pixel = image.buffer[ny * image.width + nx];
                        if (kernel_pixel.r == 0) {
                            erosion = true;                        
                        }
                    }
                }
            }
            // std::cout << "passez la boucle" << std::endl;
            if (erosion) {

                //std::cout << "(y: " << y << ", x: " << x << ")" << std::endl;
                rgb8& pixel = copy.buffer[y * copy.width + x];
                pixel.r = 0;
                pixel.g = 0;
                pixel.b = 0;
            }
        }
        image = copy;  // Appliquer la copie modifiée à l'image d'origine
    }
}

// Appliquer une opération de dilatation
void dilate(ImageView<rgb8>& image, const std::vector<std::vector<int>>& kernel, int radius) {
    ImageView<rgb8> copy = image;  // Faire une copie temporaire de l'image pour éviter la corruption
    int diameter = 2 * radius + 1;
    rgb8& intensity = image.buffer[0];
    for (int y = radius; y < image.height - radius; ++y) {
        for (int x = radius; x < image.width - radius; ++x) {
            bool dilatation = false;
            rgb8& pixel = image.buffer[y * image.width + x];
            if(pixel.r != 0){
                continue;
            }
            for (int ky = 0; !dilatation && ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {
                    if (kernel[ky][kx] == 1){
                        int ny = y + ky - radius;
                        int nx = x + kx - radius;
                        rgb8 kernel_pixel = image.buffer[ny * image.width + nx];
                        if (kernel_pixel.r != 0) {
                            dilatation = true;
                            intensity = kernel_pixel;
                        }
                    }
                }
            }
            if (dilatation) {
                rgb8& pixel = copy.buffer[y * copy.width + x];
                pixel.r = intensity.r;
                pixel.g = intensity.g;
                // pixel.b = 0;
            }
        }
    }
    
    image = copy;  // Appliquer la copie modifiée à l'image d'origine
}

// Ouverture morphologique (érosion suivie de dilatation)
void morphologicalOpening(ImageView<rgb8>& image, int radius) {
    // std::cout << "start morphologie"  << std::endl;
    int min_dimension = std::min(image.width, image.height);
    int radius = std::max(3, (min_dimension / 100) * 5); // Rayon ajusté à 1% de la taille avec un minimum de 3

    // Créer un noyau en forme de disque avec le rayon calculé
    auto diskKernel = createDiskKernel(radius);
    // Étape 1 : Erosion
    // erode(image,  radius);
    // std::cout << "dilatation" << std::endl;
    // Étape 2 : Dilatation
    dilate(image, diskKernel, radius);
}

// // seuillage d'hystérésis
// ImageView<rgb8> HysteresisThreshold(ImageView<rgb8> in) {
//   const int lowThreshold = 10; 
//   const int highThreshold = 65;

//   for (int y = 0; y < in.height; y++) {
//     for (int x = 0; x < in.width; x++) {
//       int index = y * in.width + x;
//       rgb8 pixel = in.buffer[index];

//       int intensity = (pixel.r + pixel.g + pixel.b) / 3;

//       if (intensity >= highThreshold) {
//         in.buffer[index] = {255, 255, 255};
//       } 
//       else if (intensity >= lowThreshold) {
//         bool connectedToStrongEdge = false;

//         for (int dy = -1; dy <= 1; dy++) {
//           for (int dx = -1; dx <= 1; dx++) {
//             if (dy == 0 && dx == 0) continue; // Ignore le pixel actuel
//             int neighborX = x + dx;
//             int neighborY = y + dy;

//             if (neighborX >= 0 && neighborX < in.width && neighborY >= 0 && neighborY < in.height) {
//               int neighborIndex = neighborY * in.width + neighborX;
//               rgb8 neighborPixel = in.buffer[neighborIndex];
//               int neighborIntensity = (neighborPixel.r + neighborPixel.g + neighborPixel.b) / 3;

//               if (neighborIntensity >= highThreshold) {
//                 connectedToStrongEdge = true;
//                 break;
//               }
//             }
//           }
//           if (connectedToStrongEdge) break;
//         }

//         if (connectedToStrongEdge) {
//           in.buffer[index] = {255, 255, 255};
//         } else {
//           in.buffer[index] = {0, 0, 0};
//         }
//       } 
//       else {
//         in.buffer[index] = {0, 0, 0};
//       }
//     }
//   }

//   return in;
// }

ImageView<rgb8> HysteresisThreshold(ImageView<rgb8> in) {
  const int lowThreshold = 10; 
  const int highThreshold = 35;

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

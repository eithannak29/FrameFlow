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
  const int lowThreshold = 20; 
  const int highThreshold = 45;

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

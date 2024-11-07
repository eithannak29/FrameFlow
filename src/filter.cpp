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

// seuillage d'hystérésis

ImageView<rgb8> HysteresisThreshold(ImageView<rgb8> in, const std::vector<double>& distances) {
  // Seuil bas et seuil haut pour le seuillage d'hystérésis
  const double lowThreshold = 0.35;     // Seuil bas pour contour faible
  const double highThreshold = 1;    // Seuil haut pour contour fort

  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;
      double distance = distances[index];

      // Seuillage d'hystérésis
      if (distance >= highThreshold) {
        // Marqué directement comme contour fort (blanc)
        in.buffer[index] = {255, 255, 255};
      } 
      else if (distance >= lowThreshold) {
        // Contour faible : vérifie les voisins pour continuité
        bool connectedToStrongEdge = false;
        
        // Vérifie les pixels voisins pour voir si l'un est un contour fort
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            if (dy == 0 && dx == 0) continue; // Ignore le pixel actuel
            int neighborX = x + dx;
            int neighborY = y + dy;
            
            if (neighborX >= 0 && neighborX < in.width && neighborY >= 0 && neighborY < in.height) {
              int neighborIndex = neighborY * in.width + neighborX;
              if (distances[neighborIndex] >= highThreshold) {
                connectedToStrongEdge = true;
                break;
              }
            }
          }
          if (connectedToStrongEdge) break;
        }

        // Si connecté à un contour fort, on marque en blanc, sinon en noir
        if (connectedToStrongEdge) {
          in.buffer[index] = {255, 255, 255};
        } else {
          in.buffer[index] = {0, 0, 0};
        }
      } 
      else {
        // Considéré comme arrière-plan (noir) si sous le seuil bas
        in.buffer[index] = {0, 0, 0};
      }
    }
  }
  
  return in;
}
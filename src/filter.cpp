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

// Appliquer une hystérésis pour conserver les bords forts et ceux connectés
void hysteresis(ImageView<rgb8>& image, int lowThreshold, int highThreshold) {
    int width = image.width;
    int height = image.height;
    std::vector<bool> visited(width * height, false); // Pour éviter les doublons
    std::queue<std::pair<int, int>> edgePixels; // Pixels candidats pour la propagation

    // Parcourir l'image et appliquer le seuil élevé pour détecter les bords forts
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * image.stride + x;
            rgb8& pixel = image.buffer[index];

            // Si le pixel est un bord fort
            if (pixel.r > highThreshold && !visited[index]) {
                edgePixels.push({x, y});
                visited[index] = true;
            } else if (pixel.r < lowThreshold) {
                // Pixels en dessous du seuil bas sont supprimés
                pixel.r = 0;
                pixel.g = 0;
                pixel.b = 0;
            }
        }
    }

    // Propagation des bords faibles connectés aux bords forts
    std::vector<std::pair<int, int>> directions = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    while (!edgePixels.empty()) {
        auto [x, y] = edgePixels.front();
        edgePixels.pop();

        for (auto [dx, dy] : directions) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nIndex = ny * image.stride + nx;
                rgb8& neighborPixel = image.buffer[nIndex];

                // Vérifie si le voisin est un bord faible et non visité
                if (neighborPixel.r >= lowThreshold && !visited[nIndex]) {
                    visited[nIndex] = true;
                    edgePixels.push({nx, ny});
                } else if (neighborPixel.r < lowThreshold) {
                    // Supprime les pixels en dessous du seuil bas
                    neighborPixel.r = 0;
                    neighborPixel.g = 0;
                    neighborPixel.b = 0;
                }
            }
        }
    }
}

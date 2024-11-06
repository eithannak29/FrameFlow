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

// Fonction pour appliquer un seuillage d'hystérésis sur les pixels en mouvement
void hysteresisThresholding(ImageView<rgb8>& in, const ImageView<rgb8>& bg, double lowThreshold, double highThreshold) {
    int width = in.width;
    int height = in.height;

    // Créer une copie de l'image d'entrée pour marquer les pixels visités
    std::vector<bool> strongEdges(width * height, false);
    std::vector<bool> weakEdges(width * height, false);

    // Première passe : détection des pixels forts et faibles
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            rgb8 pixel = in.buffer[index];
            rgb8 bg_pixel = bg.buffer[index];

            // Calculer la distance de couleur entre le pixel actuel et le fond
            int dr = pixel.r - bg_pixel.r;
            int dg = pixel.g - bg_pixel.g;
            int db = pixel.b - bg_pixel.b;
            double distance = std::sqrt(dr * dr + dg * dg + db * db);

            // Appliquer les seuils d'hystérésis
            if (distance >= highThreshold) {
                strongEdges[index] = true;
            } else if (distance >= lowThreshold) {
                weakEdges[index] = true;
            }
        }
    }

    // Deuxième passe : propagation des pixels faibles connectés aux pixels forts
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            if (strongEdges[index]) {
                // Marquer les pixels faibles connectés à ce pixel fort
                std::queue<int> neighbors;
                neighbors.push(index);

                while (!neighbors.empty()) {
                    int currentIndex = neighbors.front();
                    neighbors.pop();

                    int currentY = currentIndex / width;
                    int currentX = currentIndex % width;

                    // Parcourir les pixels voisins
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int neighborX = currentX + dx;
                            int neighborY = currentY + dy;
                            int neighborIndex = neighborY * width + neighborX;

                            if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height &&
                                weakEdges[neighborIndex] && !strongEdges[neighborIndex]) {
                                strongEdges[neighborIndex] = true;
                                neighbors.push(neighborIndex);
                            }
                        }
                    }
                }
            }
        }
    }

    // Troisième passe : mettre à zéro les pixels qui ne font pas partie des "strongEdges"
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            if (!strongEdges[index]) {
                in.buffer[index] = {0, 0, 0};
            }
        }
    }
}

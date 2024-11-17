#ifndef UTILS_HPP
#define UTILS_HPP

#include "Image.hpp"

// Structure pour représenter une couleur en espace Lab
struct Lab
{
    double L;
    double a;
    double b;
};

// Fonctions de conversion de couleur et de distance de couleur
double sRGBToLinear(double c);
double f_xyz_to_lab(double t);
void rgbToXyz(const rgb8& rgb, double& X, double& Y, double& Z);
Lab xyzToLab(double X, double Y, double Z);
Lab rgbToLab(const rgb8& rgb);
double deltaE(const Lab& lab1, const Lab& lab2);

// Fonctions pour le traitement d'images
double matchImagesLab(const ImageView<rgb8>& img1, const ImageView<rgb8>& img2);
void average(ImageView<rgb8>& img1, const ImageView<rgb8>& img2,
             double adaptationRate);
template <class T>
std::vector<T> saveInitialBuffer(const T* sourceBuffer, int width, int height);

// Fonction pour mapper une valeur entre 0 et 1 à une couleur RGB (carte
// thermique)
rgb8 mapToHeatmap(double value);

#endif // utils_UTILS_HPP

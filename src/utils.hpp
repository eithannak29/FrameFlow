#ifndef UTILS_HPP
#define UTILS_HPP

#include "Image.hpp"
#include <vector>
#include <tuple>

// Structure pour représenter une couleur en espace Lab
struct Lab {
    double L;
    double a;
    double b;
};

// Variables globales
extern ImageView<rgb8> bg_value;
extern ImageView<rgb8> candidate_value;
extern int time_since_match;
extern bool initialized;

// Fonctions de conversion de couleur et de distance de couleur
double sRGBToLinear(double c);
double f_xyz_to_lab(double t);
void rgbToXyz(const rgb8& rgb, double& X, double& Y, double& Z);
Lab xyzToLab(double X, double Y, double Z);
Lab rgbToLab(const rgb8& rgb);
double deltaE(const Lab& lab1, const Lab& lab2);

// Fonctions pour le traitement d'images
void init_background_model(ImageView<rgb8> in);
ImageView<rgb8> applyFilter(ImageView<rgb8> in, std::vector<double> distances);
std::tuple<double, std::vector<double>> matchImagesLab(const ImageView<rgb8>& img1, const ImageView<rgb8>& img2);
void average(ImageView<rgb8>& img1, const ImageView<rgb8> img2);
//ImageView<rgb8> applyFilterHeatmap(ImageView<rgb8> in, const std::vector<double>& distances);

// Fonction pour mapper une valeur entre 0 et 1 à une couleur RGB (carte thermique)
rgb8 mapToHeatmap(double value);

#endif // COMPUTE_UTILS_UTILS_HPP
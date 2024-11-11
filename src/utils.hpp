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
// extern Image<rgb8> bg_value;
// extern Image<rgb8> candidate_value;
// extern ImageView<uint8_t> time_since_match;

double sRGBToLinear(double c);
double f_xyz_to_lab(double t);
void rgbToXyz(const rgb8& rgb, double& X, double& Y, double& Z);
Lab xyzToLab(double X, double Y, double Z);
Lab rgbToLab(const rgb8& rgb);
double deltaE(const Lab& lab1, const Lab& lab2);

void init_background_model(ImageView<rgb8> in);
void image_copy(ImageView<rgb8> in, ImageView<rgb8> copy);

#endif // COMPUTE_UTILS_UTILS_HPP

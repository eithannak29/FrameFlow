#ifndef UTILS_HPP
#define UTILS_HPP

#include "Image.hpp"

// Structure pour représenter une couleur en espace Lab
struct Lab {
    double L;
    double a;
    double b;
};

double sRGBToLinear(double c);
double f_xyz_to_lab(double t);
void rgbToXyz(const rgb8& rgb, double& X, double& Y, double& Z);
Lab xyzToLab(double X, double Y, double Z);
Lab rgbToLab(const rgb8& rgb);
double deltaE(const Lab& lab1, const Lab& lab2);

#endif // COMPUTE_UTILS_UTILS_HPP

#include "Utils.hpp"
#include "Image.hpp"

#include <iostream>
#include <cmath>

#define SQR(x) ((x)*(x))

//------RGB/Lab conversion functions (values from OpenCV library)------

double sRGBToLinear(double c) {
    if (c <= 0.04045)
        return c / 12.92;
    else
        return std::pow((c + 0.055) / 1.055, 2.4);
}

// Function for XYZ to Lab conversion
double f_xyz_to_lab(double t) {
    const double epsilon = 0.008856;
    const double kappa = 903.3;

    if (t > epsilon)
        return std::cbrt(t);
    else
        return (kappa * t + 16.0) / 116.0;
}

// Function to convert RGB to XYZ
void rgbToXyz(const rgb8& rgb, double& X, double& Y, double& Z) {
    double r = sRGBToLinear(rgb.r / 255.0);
    double g = sRGBToLinear(rgb.g / 255.0);
    double b = sRGBToLinear(rgb.b / 255.0);

    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

// Function to convert XYZ to Lab
Lab xyzToLab(double X, double Y, double Z) {
    const double Xr = 0.95047;
    const double Yr = 1.00000;
    const double Zr = 1.08883;

    double x = X / Xr;
    double y = Y / Yr;
    double z = Z / Zr;

    double fx = f_xyz_to_lab(x);
    double fy = f_xyz_to_lab(y);
    double fz = f_xyz_to_lab(z);

    Lab lab;
    lab.L = 116.0 * fy - 16.0;
    lab.a = 500.0 * (fx - fy);
    lab.b = 200.0 * (fy - fz);

    return lab;
}

// Function to convert RGB to Lab
Lab rgbToLab(const rgb8& rgb) {
    double X, Y, Z;
    rgbToXyz(rgb, X, Y, Z);
    return xyzToLab(X, Y, Z);
}

// Function to compute ΔE (CIE76) between two Lab colors
double deltaE(const Lab& lab1, const Lab& lab2) {
    double dL = lab1.L - lab2.L;
    double da = lab1.a - lab2.a;
    double db = lab1.b - lab2.b;
    return std::sqrt(dL * dL + da * da + db * db);
}

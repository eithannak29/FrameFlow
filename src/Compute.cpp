#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"

#include <chrono>
#include <cmath>
#include <thread>
#include <iostream>


struct Lab{
    double L;
    double a;
    double b;
};

/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);


/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

Image<rgb8> bg_value;
Image<rgb8> candidate_value;
ImageView<uint8_t> time_since_match;
bool initialized = false;

double sRGBToLinear(double c) {
    if (c <= 0.04045)
        return c / 12.92;
    else
        return std::pow((c + 0.055) / 1.055, 2.4);
}

// Fonction auxiliaire pour la conversion XYZ -> Lab
double f_xyz_to_lab(double t) {
    const double epsilon = 0.008856; // (6/29)^3
    const double kappa = 903.3;      // (29/3)^3

    if (t > epsilon)
        return std::cbrt(t); // Racine cubique
    else
        return (kappa * t + 16.0) / 116.0;
}

// Fonction pour convertir RGB en XYZ
void rgbToXyz(const rgb8& rgb, double& X, double& Y, double& Z) {
    // Normalisation des valeurs RGB entre 0 et 1
    double r = sRGBToLinear(rgb.r / 255.0);
    double g = sRGBToLinear(rgb.g / 255.0);
    double b = sRGBToLinear(rgb.b / 255.0);

    // Matrice de conversion sRGB D65
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

// Fonction pour convertir XYZ en Lab
Lab xyzToLab(double X, double Y, double Z) {
    // Blanc de référence D65
    const double Xr = 0.95047;
    const double Yr = 1.00000;
    const double Zr = 1.08883;

    // Normalisation par rapport au blanc de référence
    double x = X / Xr;
    double y = Y / Yr;
    double z = Z / Zr;

    // Application de la fonction f(t)
    double fx = f_xyz_to_lab(x);
    double fy = f_xyz_to_lab(y);
    double fz = f_xyz_to_lab(z);

    Lab lab;
    lab.L = 116.0 * fy - 16.0;
    lab.a = 500.0 * (fx - fy);
    lab.b = 200.0 * (fy - fz);

    return lab;
}

// Fonction pour convertir RGB en Lab
Lab rgbToLab(const rgb8& rgb) {
    double X, Y, Z;
    rgbToXyz(rgb, X, Y, Z);
    return xyzToLab(X, Y, Z);
}

// Fonction pour calculer la distance ΔE (CIE76) entre deux couleurs Lab
double deltaE(const Lab& lab1, const Lab& lab2) {
    double dL = lab1.L - lab2.L;
    double da = lab1.a - lab2.a;
    double db = lab1.b - lab2.b;
    return std::sqrt(dL * dL + da * da + db * db);
}


double background_estimation(ImageView<rgb8> in, int x, int y)
{
    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
    rgb8* bg_pixel = (rgb8*)((std::byte*)bg_value.buffer + y * bg_value.stride);
    rgb8* candidate_pixel = (rgb8*)((std::byte*)candidate_value.buffer + y * candidate_value.stride);

    double distance = deltaE(rgbToLab(pixel[x]), rgbToLab(bg_pixel[x]));
    bool match = distance < 25;

    // std::cout << "aled: " << distance << std::endl;

    uint8_t *time = (uint8_t*)((std::byte*)time_since_match.buffer + y * time_since_match.stride);

    if (!match)
    {
        if(time[x] == 0)
        {
            candidate_pixel[x].r = pixel[x].r;
            candidate_pixel[x].g = pixel[x].g;
            candidate_pixel[x].b = pixel[x].b;
            time[x] += 1;
        }
        else if (time[x] < 100)
        {
            candidate_pixel[x].r = (candidate_pixel[x].r + pixel[x].r) / 2;
            candidate_pixel[x].g = (candidate_pixel[x].g + pixel[x].g) / 2;
            candidate_pixel[x].b = (candidate_pixel[x].b + pixel[x].b) / 2;
            time[x] += 1;
        }
        else
        {
            std::swap(bg_pixel[x].r, candidate_pixel[x].r);
            std::swap(bg_pixel[x].g, candidate_pixel[x].g);
            std::swap(bg_pixel[x].b, candidate_pixel[x].b);
            time[x] = 0;        
        }
    }
    else {
        bg_pixel[x].r = (bg_pixel[x].r + pixel[x].r) / 2;
        bg_pixel[x].g = (bg_pixel[x].g + pixel[x].g) / 2;
        bg_pixel[x].b = (bg_pixel[x].b + pixel[x].b) / 2; 
        time[x] = 0;
    }
    return distance;
}

void background_estimation_process(ImageView<rgb8> in)
{
    const double treshold = 25;
    const double distanceMultiplier = 2.8;

    for (int y = 0; y < in.height; ++y)
    {
        for (int x = 0; x < in.width; ++x)
        {
            // std::cout << "aled" << std::endl;
            double distance = background_estimation(in, x, y);
            // std::cout << "aled++" << std::endl;
            rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
             
            pixel[x].r = static_cast<uint8_t>(std::min(255.0, distance * distanceMultiplier));
        }
    }
}

/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
  Image<rgb8> img = Image<rgb8>();
  img.buffer = in.buffer;
  img.width = in.width;
  img.height = in.height;
  img.stride = in.stride;
  if (!initialized)
  {
      initialized = true;
      bg_value = img.clone();
      candidate_value = img.clone();
      // init_background_model(in);
      uint8_t* buffer = (uint8_t*)calloc(in.width * in.height, sizeof(uint8_t));
      time_since_match = ImageView<uint8_t>{buffer, in.width, in.height, in.width};
  }
  std::cout << "out" << std::endl;
  // Image<rgb8> copy = img.clone();
  background_estimation_process(in);
}


extern "C" {

  static Parameters g_params;

  void cpt_init(Parameters* params)
  {
    g_params = *params;
  }

  void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
  {
    auto img = ImageView<rgb8>{(rgb8*)buffer, width, height, stride};
    if (g_params.device == e_device_t::CPU)
      compute_cpp(img);
    else if (g_params.device == e_device_t::GPU)
      compute_cu(img);
  }

}

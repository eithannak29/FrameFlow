#include "Compute.hpp"
#include "Image.hpp"
//#include "logo.h"

#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iostream>

#define SQR(x) ((x)*(x))

/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);


/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

static ImageView<rgb8> bg_value;
static ImageView<rgb8> candidate_value;
static int time_since_match;
static bool initialized = false;

void init_background_model(ImageView<rgb8> in)
{
  bg_value = ImageView<rgb8>{new rgb8[in.width * in.height], in.width, in.height, in.stride};
  candidate_value = ImageView<rgb8>{new rgb8[in.width * in.height], in.width, in.height, in.stride};
  for (int y=0; y < in.height; y++){
    for (int x=0; x < in.width; x++){
      int index = y * in.width + x;
      bg_value.buffer[index] = in.buffer[index];
      candidate_value.buffer[index] = in.buffer[index];
    }
  }
  time_since_match = 0;
}

// Structure pour représenter une couleur en espace Lab
struct Lab {
    double L;
    double a;
    double b;
};

// Fonction pour convertir sRGB en RGB linéaire
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

// Fonction pour mapper une valeur entre 0 et 1 à une couleur RGB (carte thermique)
rgb8 mapToHeatmap(double value) {
    value = std::max(0.0, std::min(1.0, value));
    rgb8 color;
    if (value <= 0.5) {
        color.r = 0;
        color.g = static_cast<uint8_t>(value * 2 * 255);
        color.b = 255 - color.g;
    } else {
        color.r = static_cast<uint8_t>((value - 0.5) * 2 * 255);
        color.g = 255 - color.r;
        color.b = 0;
    }
    return color;
}

// Fonction pour générer une carte thermique de mouvement
void applyMotionHeatmap(const ImageView<rgb8>& bg, ImageView<rgb8>& in) {
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            int index = y * in.width + x;
            rgb8& current_pixel = in.buffer[index];
            rgb8& bg_pixel = bg.buffer[index];

            Lab current_lab = rgbToLab(current_pixel);
            Lab bg_lab = rgbToLab(bg_pixel);
            double deltaE_value = deltaE(current_lab, bg_lab);

            double normalized_value = std::min(deltaE_value / 100.0, 1.0);
            current_pixel = mapToHeatmap(normalized_value);
        }
    }
}


// Fonction optimisée pour calculer la distance moyenne en utilisant la distance Lab
double matchImagesLab(const ImageView<rgb8>& img1, const ImageView<rgb8>& img2) {
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "Erreur : les dimensions des images ne correspondent pas." << std::endl;
        return -1.0;  // Retourne une valeur indicative d'erreur
    }

    double totalDistance = 0.0;
    int numPixels = img1.width * img1.height;

    // Pré-calcul des strides en nombre de pixels
    int stride1 = img1.stride / sizeof(rgb8);
    int stride2 = img2.stride / sizeof(rgb8);

    for (int y = 0; y < img1.height; ++y) {
        rgb8* row1 = img1.buffer + y * stride1;
        rgb8* row2 = img2.buffer + y * stride2;

        for (int x = 0; x < img1.width; ++x) {
            rgb8& pixel1 = row1[x];
            rgb8& pixel2 = row2[x];

            // Convertir les pixels RGB en Lab
            Lab lab1 = rgbToLab(pixel1);
            Lab lab2 = rgbToLab(pixel2);

            // Calculer la distance ΔE
            double distance = deltaE(lab1, lab2);
            totalDistance += distance;
        }
    }

    double averageDistance = totalDistance / numPixels;
    return averageDistance;
}


double matchImagesRGB(ImageView<rgb8>& img1, ImageView<rgb8>& img2) {
    // std::cout << "Enter match computation"<< std::endl;
    // std::cout <<"width 1:"<<img1.width<<" 2:"<<img2.width << std::endl;
    // std::cout <<"height 1:"<<img1.height<<" 2:"<<img2.height << std::endl;
    if (img1.width != img2.width || img1.height != img2.height){
        std::cout << "Error: images dimensions"<< std::endl;
        return false;  
    }
    double totalDistance = 0;
    int numPixels = 0;

    for (int y = 0; y < img1.height; ++y) {
        for (int x = 0; x < img1.width; ++x) {
            rgb8* pixel1 = img1.buffer + y * (img1.stride / sizeof(rgb8)) + x;
            rgb8* pixel2 = img2.buffer + y * (img2.stride / sizeof(rgb8)) + x;

            // Calculer la distance Euclidienne entre les composantes RGB
            int dr = pixel1->r - pixel2->r;
            int dg = pixel1->g - pixel2->g;
            int db = pixel1->b - pixel2->b;
            double distance = std::sqrt(dr * dr + dg * dg + db * db);
            
            totalDistance += distance;
            numPixels++;
        }
    }
    // std::cout << "total distance: " << totalDistance << std::endl;
    double averageDistance = totalDistance / numPixels;
    // std::cout << "average distance " << averageDistance << std::endl;
    return averageDistance;
}


void average(ImageView<rgb8>& img1, const ImageView<rgb8> img2) {
  for (int y=0; y < img1.height; y++){
    for (int x=0; x < img1.width; x++){
      int index = y * img1.width + x * 3;
      rgb8 pixel1 = img1.buffer[index];
      rgb8 pixel2 = img2.buffer[index];

      pixel1.r = (pixel1.r + pixel2.r) / 2;
      pixel1.g = (pixel1.g + pixel2.g) / 2;
      pixel1.b = (pixel1.b + pixel2.b) / 2;
    }
  }
}

int background_estimation_process(ImageView<rgb8> in){
  double match_distance = matchImagesLab(bg_value, in);
  double treshold = 0.8;
  
  if (match_distance < treshold){
    average(bg_value, in);
    time_since_match = 0;
  }
  else{
    if (time_since_match == 0){
      for (int y=0; y < in.height; y++){
        for (int x=0; x < in.width; x++){
          int index = y * in.width + x;
          candidate_value.buffer[index] = in.buffer[index];
        }
      }
      time_since_match++;
    }
    else if (time_since_match < 3){
      average(candidate_value, in);
      time_since_match++;
    }
    else{
      std::swap(bg_value, candidate_value);
      time_since_match = 0;
    }
  }
  std::cout << "Background match distance: " << match_distance << std::endl;
  return match_distance;
}

/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
  if (!initialized)
  {
    // std::cout << "Initialized Background" << std::endl;
    init_background_model(in);
    initialized = true;
  }
  else{
    // std::cout << "Background estimation" << std::endl;
    background_estimation_process(in);
  }
  applyMotionHeatmap(bg_value, in);
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

#include "Compute.hpp"
#include "Image.hpp"
//#include "logo.h"

#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

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
  memcpy(&bg_value, &in, sizeof(long(ImageView<rgb8>)));
  memcpy(&candidate_value, &in, sizeof(long(ImageView<rgb8>)));
  time_since_match = 0;
}

void RGBtoXYZ(uint8_t R, uint8_t G, uint8_t B, double& X, double& Y, double& Z) {
    // Convertir de [0,255] à [0,1]
    double r = R / 255.0;
    double g = G / 255.0;
    double b = B / 255.0;

    // Convertir à l'espace linéaire
    r = r > 0.04045 ? pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = g > 0.04045 ? pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = b > 0.04045 ? pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    // Utiliser les coefficients pour l'espace sRGB vers XYZ
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

void XYZtoLAB(double X, double Y, double Z, double& L, double& a, double& b) {
    double xr = X / 95.047;
    double yr = Y / 100.000;
    double zr = Z / 108.883;

    xr = xr > 0.008856 ? std::pow(xr, 1.0/3.0) : (7.787 * xr) + (16.0/116.0);
    yr = yr > 0.008856 ? std::pow(yr, 1.0/3.0) : (7.787 * yr) + (16.0/116.0);
    zr = zr > 0.008856 ? std::pow(zr, 1.0/3.0) : (7.787 * zr) + (16.0/116.0);

    L = (116 * yr) - 16;
    a = 500 * (xr - yr);
    b = 200 * (yr - zr);
}

double LABDistance(double L1, double a1, double b1, double L2, double a2, double b2) {
    return std::sqrt((L2 - L1) * (L2 - L1) + (a2 - a1) * (a2 - a1) + (b2 - b1) * (b2 - b1));
    //return sqrt(SQR(rgb_a.r - rgb_b.r) + SQR(rgb_a.g - rgb_b.g) + SQR(rgb_a.b - rgb_b.b));
}

double matchImagesLAB(ImageView<rgb8>& img1, ImageView<rgb8>& img2) {
    if (img1.width != img2.width || img1.height != img2.height)
        return false;

    double totalDistance = 0;
    int numPixels = 0;

    for (int y = 0; y < img1.height; ++y) {
        for (int x = 0; x < img1.width; ++x) {
            //char* pixel1 = img1.buffer + y * img1.stride + x * 3;
            rgb8* pixel1 = img1.buffer + y * (img1.stride / sizeof(rgb8)) + x;
            //char* pixel2 = img2.buffer + y * img2.stride + x * 3;
            rgb8* pixel2 = img2.buffer + y * (img2.stride / sizeof(rgb8)) + x;

            double X1, Y1, Z1, L1, a1, b1;
            double X2, Y2, Z2, L2, a2, b2;

            RGBtoXYZ(pixel1->r, pixel1->g, pixel1->b, X1, Y1, Z1);
            XYZtoLAB(X1, Y1, Z1, L1, a1, b1);
            RGBtoXYZ(pixel2->r, pixel2->g, pixel2->b, X2, Y2, Z2);
            XYZtoLAB(X2, Y2, Z2, L2, a2, b2);

            double distance = LABDistance(L1, a1, b1, L2, a2, b2);
            totalDistance += distance;
            numPixels++;
        }
    }

    double averageDistance = totalDistance / numPixels;
    return averageDistance;
}

double matchImagesRGB(ImageView<rgb8>& img1, ImageView<rgb8>& img2) {
    std::cout << "Enter match computation"<< std::endl;
    if (img1.width != img2.width || img1.height != img2.height)
        std::cout << "mauvaise dimension d'images "<< std::endl;
        return false;  // Retourner une valeur indicative d'erreur, par exemple, -1.

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
    std::cout << "tatal distance: " << totalDistance << std::endl;
    double averageDistance = totalDistance / numPixels;
    std::cout << "average distance " << averageDistance << std::endl;
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
  double match_distance = matchImagesRGB(bg_value, in);//matchImagesLAB(bg_value, in);

  if (match_distance < 25){
    average(bg_value, in);
    time_since_match = 0;
  }
  else{
    if (time_since_match == 0){
      memcpy(&candidate_value, &in, in.width * in.height * sizeof(rgb8));
      time_since_match++;
    }
    else if (time_since_match < 100){
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
    init_background_model(in);
    initialized = true;
  }
  else{
    background_estimation_process(in);
  }
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
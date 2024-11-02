#include "Compute.hpp"
#include "Image.hpp"
//#include "logo.h"

#include <chrono>
#include <thread>
#include <vector>
#include <cmath>
#include <iostream>


/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);


/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);


struct BackgroundPixel {
    ImageView<rgb8> bg_value;
    ImageView<rgb8> candidate_value;
    int time_since_match = 0;
};

BackgroundPixel background_model;

void init_background_model(ImageView<rgb8> in)
{
    background_model = malloc(sizeof(BackgroundPixel));
    background_model.bg_value = memcpy(in.buffer, sizeof(ImageView<rgb8>));
    background_model.candidate_value = memcpy(in.buffer, sizeof(ImageView<rgb8>));
    background_model.time_since_match = calloc(sizeof(int), 0);
}

void RGBtoXYZ(int R, int G, int B, double& X, double& Y, double& Z) {
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
}

bool matchImagesLAB(ImageView<rgb8>& img1, ImageView<rgb8>& img2) {
    if (img1.width != img2.width || img1.height != img2.height)
        return false;

    double totalDistance = 0;
    int numPixels = 0;

    for (int y = 0; y < img1.height; ++y) {
        for (int x = 0; x < img1.width; ++x) {
            unsigned char* pixel1 = img1.buffer + y * img1.stride + x * 3;
            unsigned char* pixel2 = img2.buffer + y * img2.stride + x * 3;

            double X1, Y1, Z1, L1, a1, b1;
            double X2, Y2, Z2, L2, a2, b2;

            RGBtoXYZ(pixel1[0], pixel1[1], pixel1[2], X1, Y1, Z1);
            XYZtoLAB(X1, Y1, Z1, L1, a1, b1);
            RGBtoXYZ(pixel2[0], pixel2[1], pixel2[2], X2, Y2, Z2);
            XYZtoLAB(X2, Y2, Z2, L2, a2, b2);

            double distance = LABDistance(L1, a1, b1, L2, a2, b2);
            totalDistance += distance;
            numPixels++;
        }
    }

    double averageDistance = totalDistance / numPixels;
    return averageDistance;
}

void average(ImageView<rgb8>& img1, const ImageView<rgb8>& img2) {
  for (int y; y < img1.height; y++){
    for (int x; x < img1.width; x++){
      int index = y * img1.width + x * 3;
      img1.buffer[index] = (img1.buffer[index] + img2.buffer[index]) / 2;
    }
  }
}

int background_estimation_process(ImageView<rgb8> in){
  double match_distance = matchImagesLAB(background_model.bg_value, current_frame, 25)
  if (match_distance < 25){
    average(&background_model.bg_value, &current_frame);
    background_model.time_since_match = 0;
  }
  else{
    if (background_model.time_since_match == 0){
      background_model.candidate_value = memcpy(current_frame, in.width * in.height * sizeof(rgb8));
      background_model.time_since_match++;
    }
    else if (background_model.time_since_match < 100){
      average(&background_model.candidates, &background_model.current_frame);
      background_model.time_since_match++;
    }
    else{
      ImageView& tmp = memcpy(background_model.candidates, sizeof(ImageView<rgb8>));
      background_model.candidate_value = memcpy(background_model.bg_value, sizeof(ImageView<rgb8>));
      background_model.bg_value = memcpy(tmp, sizeof(ImageView<rgb8>));
      background_model.time_since_match = 0;
    }
  }
  std::cout << "Background match distance: " << match_distance << std::endl;
  return match_distance;
}

/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
  if (background_model.empty())
  {
    init_background_model(in)
  }
  else{
    background_estimation_process(in)

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
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


struct Lab {
    double L;
    double a;
    double b;
};


Lab rgb2lab(rgb8 rgb) {
    double r = rgb.r / 255.0;
    double g = rgb.g / 255.0;
    double b = rgb.b / 255.0;

    // Convert sRGB to linear RGB
    r = (r > 0.04045) ? std::pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? std::pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? std::pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    // Convert linear RGB to CIE XYZ
    double X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    double Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    double Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // Convert CIE XYZ to CIE Lab
    X /= 0.95047;
    Y /= 1.0;
    Z /= 1.08883;

    X = (X > 0.008856) ? std::pow(X, 1.0 / 3.0) : (903.3 * X + 16) / 116.0;
    Y = (Y > 0.008856) ? std::pow(Y, 1.0 / 3.0) : (903.3 * Y + 16) / 116.0;
    Z = (Z > 0.008856) ? std::pow(Z, 1.0 / 3.0) : (903.3 * Z + 16) / 116.0;

    return {
        116.0 * Y - 16.0,
        500.0 * (X - Y),
        200.0 * (Y - Z)
    };
}

double labDistance(const Lab& lab1, const Lab& lab2) {
    double dL = lab1.L - lab2.L;
    double da = lab1.a - lab2.a;
    double db = lab1.b - lab2.b;
    return std::sqrt(dL * dL + da * da + db * db);
}


double matchImagesLab(ImageView<rgb8>& img1, ImageView<rgb8>& img2) {
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cout << "Error: images dimensions" << std::endl;
        return -1.0;  // Retourne une valeur indicative d'erreur
    }
    
    double totalDistance = 0;
    int numPixels = 0;

    for (int y = 0; y < img1.height; ++y) {
        for (int x = 0; x < img1.width; ++x) {
            rgb8* pixel1 = img1.buffer + y * (img1.stride / sizeof(rgb8)) + x;
            rgb8* pixel2 = img2.buffer + y * (img2.stride / sizeof(rgb8)) + x;

            // Convertir chaque pixel RGB en Lab
            Lab lab1 = rgbToLab(*pixel1);
            Lab lab2 = rgbToLab(*pixel2);

            // Calculer la distance Lab (ΔE)
            double distance = labDistance(lab1, lab2);
            totalDistance += distance;
            numPixels++;
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
  double match_distance = matchImagesLab(bg_value, in);//matchImagesLAB(bg_value, in);

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
    std::cout << "Initialized Background" << std::endl;
    init_background_model(in);
    initialized = true;
  }
  else{
    std::cout << "Background estimation" << std::endl;
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

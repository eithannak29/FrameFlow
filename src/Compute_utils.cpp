#include "Compute_utils.hpp"
#include "Image.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cstdlib>

#define SQR(x) ((x)*(x))

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

ImageView<rgb8> applyFilter(ImageView<rgb8> in, std::vector<double> distances) {
  //const double adaptationRate = 0.1;
  const double strictDistanceThreshold = 0.25;
  const double highlightDistanceMultiplier = 2.8; 

  //ImageView<rgb8> filtered = ImageView<rgb8>{new rgb8[in.width * in.height], in.width, in.height, in.stride};

  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;
      //rgb8 pixel = in.buffer[index];
      //rgb8 bg_pixel = bg_value.buffer[index];

      double distance = distances[index];
      //std::cout << "Distance: " << distance << ", index: " << index << std::endl ;

      // Background adaptation and filtering
      if (distance < strictDistanceThreshold) {
        // Mark as background if within threshold
        in.buffer[index] = {0, 0, 0};
        
      } else {
        // For objects that differ significantly from the background, increase highlight intensity
        uint8_t intensity = static_cast<uint8_t>(std::min(255.0, distance * highlightDistanceMultiplier));
        in.buffer[index] = {intensity, intensity, 0};
      }
    }
  }
  
  return in;
}

ImageView<rgb8> applyFilterHeatmap(ImageView<rgb8> in, const std::vector<double>& distances) {
  // Calcul de la distance maximale
  double maxDistance = *std::max_element(distances.begin(), distances.end());

  // Gestion du cas où maxDistance est très faible pour éviter la division par zéro
  if (maxDistance < 1e-5) {
    maxDistance = 1e-5;
  }
  
  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;
      double distance = distances[index];
      
      // Normalisation de la distance pour un affichage en heatmap
      double normalizedDistance = distance / maxDistance;
      
      // Calcul des composantes couleur (bleu -> rouge)
      uint8_t red = static_cast<uint8_t>(255 * normalizedDistance);
      uint8_t blue = static_cast<uint8_t>(255 * (1 - normalizedDistance));
      
      // Appliquer la couleur en heatmap
      in.buffer[index] = {red, 0, blue}; 
    }
  }
  
  return in;
}

// Fonction optimisée pour calculer la distance moyenne en utilisant la distance Lab
std::tuple<double, std::vector<double>> matchImagesLab(const ImageView<rgb8>& img1, const ImageView<rgb8>& img2) {
    std::vector<double> distances;

    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "Erreur : les dimensions des images ne correspondent pas." << std::endl;
        return std::make_tuple(-1.0, distances);  // Retourne une valeur indicative d'erreur
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
            distances.push_back(distance);
        }
    }

    double averageDistance = totalDistance / numPixels;
    return std::make_tuple(averageDistance, distances);
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
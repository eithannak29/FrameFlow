#include "utils.hpp"

#include <cmath>
#include <iostream>

#include "Image.hpp"

#define SQR(x) ((x) * (x))

// Convert a value between 0 and 1 to an RGB color (heatmap)
double sRGBToLinear(double c)
{
    if (c <= 0.04045)
        return c / 12.92;
    else
        return std::pow((c + 0.055) / 1.055, 2.4);
}

// Convert XYZ to Lab
double f_xyz_to_lab(double t)
{
    const double epsilon = 0.008856; // (6/29)^3
    const double kappa = 903.3; // (29/3)^3

    if (t > epsilon)
        return std::cbrt(t); // Racine cubique
    else
        return (kappa * t + 16.0) / 116.0;
}

// Convert RGB to XYZ
void rgbToXyz(const rgb8& rgb, double& X, double& Y, double& Z)
{
    // Normalisation des valeurs RGB entre 0 et 1
    double r = sRGBToLinear(rgb.r / 255.0);
    double g = sRGBToLinear(rgb.g / 255.0);
    double b = sRGBToLinear(rgb.b / 255.0);

    // Matrice de conversion sRGB D65
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

// Convert XYZ to Lab
Lab xyzToLab(double X, double Y, double Z)
{
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

// Convert RGB to Lab
Lab rgbToLab(const rgb8& rgb)
{
    double X, Y, Z;
    rgbToXyz(rgb, X, Y, Z);
    return xyzToLab(X, Y, Z);
}

// Calculate the CIE76 ΔE distance between two Lab colors
double deltaE(const Lab& lab1, const Lab& lab2)
{
    double dL = lab1.L - lab2.L;
    double da = lab1.a - lab2.a;
    double db = lab1.b - lab2.b;
    return std::sqrt(dL * dL + da * da + db * db);
}

ImageView<rgb8> applyFilter(ImageView<rgb8> in)
{
    const double adaptationRate = 0.05;
    for (int y = 0; y < in.height; y++)
    {
        for (int x = 0; x < in.width; x++)
        {
            int index = y * in.width + x;
            rgb8 pixel = in.buffer[index];
            rgb8 bg_pixel = bg_value.buffer[index];

            // Calculer la distance de couleur entre le pixel et le fond
            int dr = pixel.r - bg_pixel.r;
            int dg = pixel.g - bg_pixel.g;
            int db = pixel.b - bg_pixel.b;
            double distance = std::sqrt(dr * dr + dg * dg + db * db);
            uint8_t intensity =
                static_cast<uint8_t>(std::min(255.0, distance * 2));

            // Appliquer un effet visuel en fonction de la distance
            if (distance < 50)
            {
                // Si la distance est faible, on met le pixel en fond
                in.buffer[index] = { 0, 0, 0 };

                bg_pixel.r =
                    static_cast<uint8_t>(bg_pixel.r * (1 - adaptationRate)
                                         + pixel.r * adaptationRate);
                bg_pixel.g =
                    static_cast<uint8_t>(bg_pixel.g * (1 - adaptationRate)
                                         + pixel.g * adaptationRate);
                bg_pixel.b =
                    static_cast<uint8_t>(bg_pixel.b * (1 - adaptationRate)
                                         + pixel.b * adaptationRate);
            }
            else
            {
                // Si la distance est élevée, on applique un effet de
                // surbrillance
                in.buffer[index] = { intensity, intensity, 0 };
            }
        }
    }
    return in;
}

// Compare two images in Lab color space
double matchImagesLab(const ImageView<rgb8>& img1, const ImageView<rgb8>& img2)
{
    if (img1.width != img2.width || img1.height != img2.height)
    {
        std::cerr << "Erreur : les dimensions des images ne correspondent pas."
                  << std::endl;
        return -1.0; // Retourne une valeur indicative d'erreur
    }

    double totalDistance = 0.0;
    int numPixels = img1.width * img1.height;

    // Pré-calcul des strides en nombre de pixels
    int stride1 = img1.stride / sizeof(rgb8);
    int stride2 = img2.stride / sizeof(rgb8);

    for (int y = 0; y < img1.height; ++y)
    {
        rgb8* row1 = img1.buffer + y * stride1;
        rgb8* row2 = img2.buffer + y * stride2;

        for (int x = 0; x < img1.width; ++x)
        {
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

// Average two images with a given adaptation rate
void average(ImageView<rgb8>& img1, const ImageView<rgb8>& img2,
             double adaptationRate)
{
    for (int y = 0; y < img1.height; y++)
    {
        for (int x = 0; x < img1.width; x++)
        {
            int index = y * img1.width + x;
            img1.buffer[index].r =
                static_cast<uint8_t>(img1.buffer[index].r * (1 - adaptationRate)
                                     + img2.buffer[index].r * adaptationRate);
            img1.buffer[index].g =
                static_cast<uint8_t>(img1.buffer[index].g * (1 - adaptationRate)
                                     + img2.buffer[index].g * adaptationRate);
            img1.buffer[index].b =
                static_cast<uint8_t>(img1.buffer[index].b * (1 - adaptationRate)
                                     + img2.buffer[index].b * adaptationRate);
        }
    }
}

template <class T>
std::vector<T> saveInitialBuffer(const T* sourceBuffer, int width, int height)
{
    int totalSize = width * height; // Nombre total de pixels
    std::vector<T> pixelArray(
        totalSize); // Création du tableau avec la taille appropriée
    std::copy(sourceBuffer, sourceBuffer + totalSize,
              pixelArray.begin()); // Copie des pixels
    return pixelArray;
}

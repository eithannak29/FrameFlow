#include "Compute.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "Image.hpp"
#include "filter.hpp"
#include "logo.h"
#include "utils.hpp"

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
double total_time_elapsed = 0.0;

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

double background_estimation(ImageView<rgb8> in, int x, int y)
{
    // rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

    rgb8* bg_pixel = (rgb8*)((std::byte*)bg_value.buffer + y * bg_value.stride);
    rgb8* candidate_pixel = (rgb8*)((std::byte*)candidate_value.buffer
                                    + y * candidate_value.stride);

    // Moyenne locale sur les pixels voisins pour une estimation plus stable
    int sumR = 0, sumG = 0, sumB = 0, count = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < in.width && ny >= 0 && ny < in.height)
            {
                rgb8 neighbor_pixel =
                    *((rgb8*)((std::byte*)in.buffer + ny * in.stride) + nx);
                sumR += neighbor_pixel.r;
                sumG += neighbor_pixel.g;
                sumB += neighbor_pixel.b;
                count++;
            }
        }
    }
    rgb8 mean_pixel = { static_cast<uint8_t>(sumR / count),
                        static_cast<uint8_t>(sumG / count),
                        static_cast<uint8_t>(sumB / count) };

    // Calcul de la distance ΔE entre le pixel de fond et la moyenne locale
    double distance = deltaE(rgbToLab(mean_pixel), rgbToLab(bg_pixel[x]));
    bool match = distance < 5;

    uint8_t* time = (uint8_t*)((std::byte*)time_since_match.buffer
                               + y * time_since_match.stride);

    if (!match)
    {
        if (time[x] == 0)
        {
            candidate_pixel[x] = mean_pixel;
            time[x] += 1;
        }
        else if (time[x] < 100)
        {
            candidate_pixel[x].r = (candidate_pixel[x].r + mean_pixel.r) / 2;
            candidate_pixel[x].g = (candidate_pixel[x].g + mean_pixel.g) / 2;
            candidate_pixel[x].b = (candidate_pixel[x].b + mean_pixel.b) / 2;
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
    else
    {
        // Mise à jour progressive du fond avec interpolation pour un lissage
        bg_pixel[x].r =
            static_cast<uint8_t>(bg_pixel[x].r * 0.9 + mean_pixel.r * 0.1);
        bg_pixel[x].g =
            static_cast<uint8_t>(bg_pixel[x].g * 0.9 + mean_pixel.g * 0.1);
        bg_pixel[x].b =
            static_cast<uint8_t>(bg_pixel[x].b * 0.9 + mean_pixel.b * 0.1);
        time[x] = 0;
    }

    return distance;
}

void background_estimation_process(ImageView<rgb8> in)
{
    const double distanceMultiplier = 2.8;

    for (int y = 0; y < in.height; ++y)
    {
        for (int x = 0; x < in.width; ++x)
        {
            // std::cout << "aled" << std::endl;
            double distance = background_estimation(in, x, y);
            // std::cout << "aled++" << std::endl;
            rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

            pixel[x].r = static_cast<uint8_t>(
                std::min(255.0, distance * distanceMultiplier));
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
        uint8_t* buffer =
            (uint8_t*)calloc(in.width * in.height, sizeof(uint8_t));
        time_since_match =
            ImageView<uint8_t>{ buffer, in.width, in.height, in.width };
    }

    std::vector<rgb8> initialPixels =
        saveInitialBuffer(in.buffer, in.width, in.height);

    // ---Background estimation---
    background_estimation_process(in);
    // ---Morphological opening---
    morphologicalOpening(in, 3);
    // ---Hysteresis thresholding---
    ImageView<rgb8> mask = HysteresisThreshold(in);
    // ---Red mask---
    in = applyRedMask(in, mask, initialPixels);
}

extern "C"
{
    static Parameters g_params;

    static int frame_counter_bench = 0;

    void cpt_init(Parameters* params)
    {
        g_params = *params;
    }

    void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
    {
        auto img = ImageView<rgb8>{ (rgb8*)buffer, width, height, stride };
        auto start = std::chrono::high_resolution_clock::now();

        if (g_params.device == e_device_t::CPU)
            compute_cpp(img);
        else if (g_params.device == e_device_t::GPU)
            compute_cu(img);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        total_time_elapsed += elapsed.count();
        frame_counter_bench++;

        // if (frame_counter_bench == FRAMES)
        // {
        //     std::string device_type =
        //         (g_params.device == e_device_t::CPU) ? "CPU" : "GPU";
        //     std::cout << "Total time " << device_type << ": "
        //               << total_time_elapsed << "s" << std::endl;
        // }
    }
}

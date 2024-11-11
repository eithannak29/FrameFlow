#include "Compute.hpp"
#include "utils.hpp"
#include "Image.hpp"
#include "logo.h"

// #include <chrono>
#include <iostream>
// #include <stdlib.h>
// #include <thread>


/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);


/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);


Image<rgb8> bg_value;
Image<rgb8> candidate_value;
ImageView<uint8_t> time_since_match;

const int FRAMES = 268;
int frame_counter = 0;

void show_progress(int current, int total){
    int barWidth = 50;
    float progress = (float) current / total;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i){
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

int index(int x, int y, int stride){
    return y * stride + x;
}

double background_estimation(ImageView<rgb8> in, int x, int y)
{
    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.width);
    rgb8* bg_pixel = (rgb8*)((std::byte*)bg_value.buffer + y * bg_value.width);
    rgb8* candidate_pixel = (rgb8*)((std::byte*)candidate_value.buffer + y * candidate_value.width);

    double distance = deltaE(rgbToLab(pixel[x]), rgbToLab(bg_pixel[x]));
    bool match = distance < 25;

    // std::cout << "aled: " << distance << std::endl;

    int *time = (int*)((std::byte*)time_since_match.buffer + y * time_since_match.width);

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
            double distance = background_estimation(in, x, y);
            rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.width);
            uint8_t intensity = static_cast<uint8_t>(std::min(255.0, distance * distanceMultiplier));
                pixel[x].r = intensity;
        }
    }
}

void reconstruction(ImageView<rgb8> in, ImageView<rgb8> copy)
{
    for (int y = 0; y < in.height; ++y)
    {
        for (int x = 0; x < in.width; ++x)
        {
            rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.width);
            rgb8* copy_pixel = (rgb8*)((std::byte*)copy.buffer + y * copy.width);
            if (copy_pixel[x].r != 0){
                pixel[x].r = copy_pixel[x].r;
                pixel[x].g = copy_pixel[x].g;
                pixel[x].b = copy_pixel[x].b;    
            }
        }
    }
}



/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
    show_progress(frame_counter, FRAMES);
    frame_counter++;
    Image<rgb8> img = Image<rgb8>();
    img.buffer = in.buffer;
    img.width = in.width;
    img.height = in.height;
    img.stride = in.stride;
    if (bg_value.buffer == nullptr)
    {
        std::cout << "aled" << std::endl; 
        bg_value = img.clone();
        candidate_value = img.clone();
        std::cout << "another" << std::endl;
        // init_background_model(in);
        void* buffer = calloc(in.width * in.height, sizeof(uint8_t));
        std::cout << "en vrai si je saute de ma fenêtre je meurs" << std::endl;
        time_since_match = ImageView<uint8_t>{(uint8_t*)buffer, in.width, in.height, in.stride};
    }
    std::cout << "ff" << std::endl;
    // Image<rgb8> copy = img.clone();
    std::cout << "py" << std::endl;
    background_estimation_process(in);
    // reconstruction(in, copy);
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

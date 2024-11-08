#include "Compute.hpp"
#include "Compute_utils.hpp"
#include "Image.hpp"
#include "filter.hpp"
#include <tuple>
#include <vector>
#include <cstdlib>
#include <cstring> // Pour std::memcpy

#include <iostream>

/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);


/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

// Global variables
ImageView<rgb8> bg_value;
ImageView<rgb8> candidate_value;
int time_since_match;
bool initialized = false;

template <typename T>
void mySwap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// Function to initialize the background model
std::tuple<double, std::vector<double>> background_estimation_process(ImageView<rgb8> in){
  auto [match_distance, distances] = matchImagesLab(bg_value, in);
  auto [match_distance_candidate, distances_candidate] = matchImagesLab(candidate_value, in); // distance entre le candidat et l'image actuelle, à voir
  double treshold = 0.25;
  
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
    else if (time_since_match < 150){
      average(candidate_value, in);
      time_since_match++;
    }
    else{
      if (match_distance_candidate < treshold){  // cas ou l'on veux changer de background, et que le mouvement n'est pas éfemère
        mySwap(bg_value, candidate_value);
        time_since_match = 0;
      }
    }
  }
  // std::cout << "Background match distance: " << match_distance << std::endl;
  return std::make_tuple(match_distance, distances);
}

template <class T>
std::vector<T> saveInitialBuffer(const T* sourceBuffer, int width, int height) {
    int totalSize = width * height; // Nombre total de pixels
    std::vector<T> pixelArray(totalSize); // Création du tableau avec la taille appropriée
    std::copy(sourceBuffer, sourceBuffer + totalSize, pixelArray.begin()); // Copie des pixels
    return pixelArray;
}

/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
  static std::vector<rgb8> initialPixels;

  if (!initialized)
  {
    // std::cout << "Initialized Background" << std::endl;
    init_background_model(in);
    initialized = true;
  }
  else{
    initialPixels = saveInitialBuffer(in.buffer, in.width, in.height);

    auto [match_distance, distances] = background_estimation_process(in);
    //std::cout << "filter" << std::endl;
    //in = applyFilter(in, distances);
    in = applyFilterHeatmap(in, distances);
    
    //in = applyFilterHeatmap(in, distances);
    //std::cout << "morphologie" << std::endl;
    morphologicalOpening(in, 3);
    //std::cout << "mask" << std::endl;
    ImageView<rgb8> mask = HysteresisThreshold(in);
    //std::cout << "apply mask" << std::endl;

    in = applyRedMask(in, mask, initialPixels);
    //std::cout << "end" << std::endl;

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

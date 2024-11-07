#include "Compute.hpp"
#include "Compute_utils.hpp"
#include "Image.hpp"
#include "filter.hpp"
#include <tuple>
#include <vector>

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
  auto [match_distance_candidate, distances_candidate] = matchImagesLab(candidate_value, in);
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
    else if (time_since_match < 100){
      average(candidate_value, in);
      time_since_match++;
    }
    else{
      if (match_distance_candidate > treshold){
        mySwap(bg_value, candidate_value);
        time_since_match = 0;
      }
    }
  }
  // std::cout << "Background match distance: " << match_distance << std::endl;
  return std::make_tuple(match_distance, distances);
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
    auto [match_distance, distances] = background_estimation_process(in);
    in = applyFilter(in, distances);
    //in = applyFilter(in, distance);
  }
  //in = applyFilter(in);
  //morphologicalOpening(in, 3);

  //hysteresisThresholding(in, bg_value, 6, 30);
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

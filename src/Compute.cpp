#include "Compute.hpp"
#include "Compute_utils.hpp"
#include "Image.hpp"
#include "filter.hpp"

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


// Function to initialize the background model
int background_estimation_process(ImageView<rgb8> in) {
  static double recent_average_distance = 0.2; // Initialize with a base threshold
  const double base_threshold = 0.2;
  const double alpha = 0.05;
  const double high_adaptation_rate = 0.1;
  const double low_adaptation_rate = 0.05;
  const double swap_threshold = 0.1;
  const int stable_frame_threshold = 5;
  static int confidence_counter = 0;
  const int confidence_limit = 3;

  double match_distance = matchImagesLab(bg_value, in);
  
  // Update the adaptive threshold based on recent average distance
  recent_average_distance = (1 - alpha) * recent_average_distance + alpha * match_distance;
  double adaptive_threshold = base_threshold + alpha * recent_average_distance;
  
  // Adaptation rate based on stability
  double adaptationRate = (time_since_match > stable_frame_threshold) ? high_adaptation_rate : low_adaptation_rate;

  if (match_distance < adaptive_threshold) {
    average(bg_value, in, adaptationRate);
    time_since_match = 0;
    confidence_counter = 0; // Reset confidence counter on stable background
  } else {
    if (time_since_match == 0) {
      // Initialize candidate background with the current frame
      for (int y = 0; y < in.height; y++) {
        for (int x = 0; x < in.width; x++) {
          int index = y * in.width + x;
          candidate_value.buffer[index] = in.buffer[index];
        }
      }
      time_since_match++;
    } else if (time_since_match < stable_frame_threshold) {
      // Blend candidate background over several frames
      average(candidate_value, in, adaptationRate);
      time_since_match++;
    } else {
      // Confidence check before swapping
      if (matchImagesLab(candidate_value, bg_value) < swap_threshold) {
        average(bg_value, candidate_value, adaptationRate);
      } else if (confidence_counter >= confidence_limit) {
        std::swap(bg_value, candidate_value);
        time_since_match = 0;
        confidence_counter = 0;

        // Clear candidate buffer
        for (int y = 0; y < candidate_value.height; y++) {
          for (int x = 0; x < candidate_value.width; x++) {
            int index = y * candidate_value.width + x;
            candidate_value.buffer[index] = {0, 0, 0};
          }
        }
      } else {
        confidence_counter++;
      }
    }
  }
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
  in = applyFilter(in);
  morphologicalOpening(in, 3);
  hysteresis(in, 50, 100);
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

#include "Image.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>

std::vector<std::vector<int>> createDiskKernel(int radius) {
    int center = radius;
    int diameter = 2 * radius + 1;

    std::vector<std::vector<int>> kernel(diameter, std::vector<int>(diameter, 0));

    for (int i = 0; i < diameter; ++i) {
        for (int j = 0; j < diameter; ++j) {
            if (std::sqrt((i - center) * (i - center) + (j - center) * (j - center)) <= radius) {
                kernel[i][j] = 1;
            }
            else
            {
                kernel[i][j] = 0;
            }
        }
    }
    return kernel;
}

Image<rgb8> clone(ImageView<rgb8> in)
{
    Image<rgb8> img = Image<rgb8>();
    img.buffer = in.buffer;
    img.width = in.width;
    img.height = in.height;
    img.stride = in.width;
    return img.clone();
}

// Function to perform morphological operations
void morphological(ImageView<rgb8> in, const std::vector<std::vector<int>>& kernel, int radius, bool erode) {
    Image<rgb8> copy = clone(in);
    int diameter = 2 * radius + 1;
    
    for (int y = radius; y < in.height - radius; ++y) {
        rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);
        for (int x = radius; x < in.width - radius; ++x) {
            
            uint8_t new_value = (erode) ? 255 : 0;
            for (int ky = 0; ky < diameter; ++ky) {
                for (int kx = 0; kx < diameter; ++kx) {
                    if (kernel[ky][kx] == 1) {
                        int ny = y + ky - radius;
                        int nx = x + kx - radius;

                        rgb8 kernel_pixel = in.buffer[ny * in.width + nx];
                        if (erode) {
                            new_value = std::min(new_value, kernel_pixel.r);
                        } else {
                            new_value = std::max(new_value, kernel_pixel.r);
                        }                     
                    }
                }
          pixel[x].r = new_value;
          }
        }
    in = copy;  
    }
}

// Apply morphological opening (erode + dilate)
void morphologicalOpening(ImageView<rgb8> in, int minradius) {
    int min_dimension = std::min(in.width, in.height);
      int ratio_disk = 2;
    int radius = (min_dimension / 100) / ratio_disk;
    auto diskKernel = createDiskKernel(radius);
    
    // Erosion
    morphological(in, diskKernel, radius, true);
    // Dilation
    morphological(in, diskKernel, radius, false);
}

void hysteresis_threshold_process(ImageView<rgb8> in, int lowThreshold, int highThreshold) {
  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;
      rgb8 pixel = in.buffer[index];
      int intensity = pixel.r;

      if (intensity >= highThreshold) {
        in.buffer[index] = {255, 255, 255};
      } else if (intensity < lowThreshold) {
        in.buffer[index] = {0, 0, 0};
      } else {
        in.buffer[index] = {127, 127, 127};
      }
    }
  }
}

void propagate_edges(ImageView<rgb8> in, int lowThreshold, int highThreshold, bool* hasChanged) {
  for (int y = 0; y < in.height; y++) {
      for (int x = 0; x < in.width; x++) {

      if (x >= in.width || y >= in.height)
          return;

      rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

      if (pixel[x].r == 255) {
          int nb_neighbors = 0;
          for (int dy = -1; dy <= 1; dy++) {
              for (int dx = -1; dx <= 1; dx++) {
                  if (dx == 0 && dy == 0) continue;

                  int neighborX = x + dx;
                  int neighborY = y + dy;

                  if (neighborX >= 0 && neighborX < in.width && neighborY >= 0 && neighborY < in.height) {
                      rgb8* neighborPixel = (rgb8*)((std::byte*)in.buffer + neighborY * in.stride);
                      int neighborIntensity = neighborPixel[neighborX].r;
                      if (neighborIntensity > lowThreshold && neighborIntensity < highThreshold && neighborPixel[neighborX].r == 127) {
                          neighborPixel[neighborX] = {255, 255, 255};
                          *hasChanged = true;
                      }
                      if (neighborIntensity >= lowThreshold) {
                          nb_neighbors++;
                      }
                  }
              }
          }
          if (nb_neighbors < 4) {
              pixel[x] = {0, 0, 0};
              *hasChanged = true;
          }
      }
    }
  }
}

ImageView<rgb8> hysteresisThreshold(ImageView<rgb8> in, int lowThreshold, int highThreshold) {
    hysteresis_threshold_process(in, lowThreshold, highThreshold);

    bool updated;
    do {
        updated = false;
        propagate_edges(in, lowThreshold, highThreshold, &updated);
    } while (updated);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= in.width || y >= in.height)
        return;

    rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

    if (pixel[x].r == 127) {
        pixel[x].r = 0;
        pixel[x].g = 0;
        pixel[x].b = 0;
    }
    return in;
}

// // Apply morphological threshold
// ImageView<rgb8> HysteresisThreshold(ImageView<rgb8> in) {
//   const int lowThreshold = 25; 
//   const int highThreshold = 50;

//   std::queue<std::pair<int, int>> edgeQueue;

//   for (int y = 0; y < in.height; y++) {
//     for (int x = 0; x < in.width; x++) {
//       int index = y * in.width + x;
//       rgb8 pixel = in.buffer[index];
//       int intensity = pixel.r;

//       if (intensity >= highThreshold) {
//         in.buffer[index] = {255, 255, 255};
//       } else if (intensity < lowThreshold) {
//         in.buffer[index] = {0, 0, 0};
//       } else {
//         in.buffer[index] = {127, 127, 127};
//       }
//     }
//   }

//   bool hasChanged = true;
//   while (hasChanged) {
//     hasChanged = false;
//     for (int y = 0; y < in.height; y++) {
//       for (int x = 0; x < in.width; x++) {
//         int index = y * in.width + x;
//         rgb8* pixel = (rgb8*)((std::byte*)in.buffer + y * in.stride);

//         if (pixel[x].r == 255) {
//           int nb_neighbors = 0;
//           for (int dy = -1; dy <= 1; dy++) {
//               for (int dx = -1; dx <= 1; dx++) {
//                   if (dx == 0 && dy == 0) continue;

//                   int neighborX = x + dx;
//                   int neighborY = y + dy;

//                   if (neighborX >= 0 && neighborX < in.width && neighborY >= 0 && neighborY < in.height) {
//                       rgb8* neighborPixel = (rgb8*)((std::byte*)in.buffer + neighborY * in.stride);
//                       int neighborIntensity = neighborPixel[neighborX].r;
//                       if (neighborIntensity > lowThreshold && neighborIntensity < highThreshold && neighborPixel[neighborX].r == 127) {
//                           neighborPixel[neighborX] = {255, 255, 255};
//                           hasChanged = true;
//                       }
//                       if (neighborIntensity >= lowThreshold) {
//                           nb_neighbors++;
//                       }
//                   }
//               }
//           }
//           if (nb_neighbors < 4) {
//               pixel[x] = {0, 0, 0};
//               hasChanged = true;
//           }
//         }
//       }
//     }
//   }


//   for (int y = 0; y < in.height; y++) {
//     for (int x = 0; x < in.width; x++) {
//       int index = y * in.width + x;
//       rgb8& pixel = in.buffer[index];

//       if (pixel.r == 127 && pixel.g == 127 && pixel.b == 127) {
//         pixel = {0, 0, 0};
//       }
//     }
//   }

//   return in;
// }

// Apply red mask
ImageView<rgb8> applyRedMask(ImageView<rgb8> in, const ImageView<rgb8>& mask, std::vector<rgb8> initialPixels) {
  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;

      if (mask.buffer[index].r > 0) {
        in.buffer[index].r = std::min(255, static_cast<int>(initialPixels[index].r + 0.5 * 255));
        in.buffer[index].g = initialPixels[index].g;
        in.buffer[index].b = initialPixels[index].b;
      }
      else {
        in.buffer[index] = initialPixels[index];
      }
    }
  }

  return in;
}

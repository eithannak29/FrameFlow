#include "Image.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>

std::vector<std::vector<int>> createDiskKernel(int radius) {
    int diameter = 2 * radius + 1;
    std::vector<std::vector<int>> kernel(diameter, std::vector<int>(diameter, 0));
    int center = radius;
    
    for (int i = 0; i < diameter; ++i) {
        for (int j = 0; j < diameter; ++j) {
            if (std::sqrt((i - center) * (i - center) + (j - center) * (j - center)) <= radius) {
                kernel[i][j] = 1;
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

                        rgb8* kernel_pixel = (rgb8*)((std::byte*)in.buffer + ny * in.stride);
                        if (erode) {
                            new_value = std::min(new_value, kernel_pixel[nx].r);
                        } else {
                            new_value = std::max(new_value, kernel_pixel[nx].r);
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
    int ratio_disk = 1;
    int radius = std::max(minradius, (min_dimension / 100) * ratio_disk);
    auto diskKernel = createDiskKernel(radius);
    
    // Erosion
    morphological(in, diskKernel, radius, true);
    // Dilation
    morphological(in, diskKernel, radius, false);
}

// Apply morphological threshold
ImageView<rgb8> HysteresisThreshold(ImageView<rgb8> in) {
  const int lowThreshold = 10; 
  const int highThreshold = 100;

  std::queue<std::pair<int, int>> edgeQueue;

  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;
      rgb8 pixel = in.buffer[index];
      int intensity = pixel.r;

      if (intensity >= highThreshold) {
        in.buffer[index] = {255, 255, 255};
        edgeQueue.push({x, y});
      } else if (intensity < lowThreshold) {
        in.buffer[index] = {0, 0, 0};
      } else {
        in.buffer[index] = {127, 127, 127};
      }
    }
  }

  while (!edgeQueue.empty()) {
    auto [x, y] = edgeQueue.front();
    int index = y * in.width + x;
    rgb8 pixel = in.buffer[index];
    edgeQueue.pop();

    int nb_neighbors = 0;
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) continue;

        int neighborX = x + dx;
        int neighborY = y + dy;

        if (neighborX >= 0 && neighborX < in.width && neighborY >= 0 && neighborY < in.height) {
          int neighborIndex = neighborY * in.width + neighborX;
          rgb8& neighborPixel = in.buffer[neighborIndex];

          int neighborIntensity = neighborPixel.r;

          if (neighborIntensity > lowThreshold && neighborIntensity < highThreshold && neighborPixel.r == 127) {
            neighborPixel = {255, 255, 255};
            edgeQueue.push({neighborX, neighborY});
          }
          if (neighborIntensity >= lowThreshold) {
            nb_neighbors++;
          }
        }
      }
    }
    if (nb_neighbors < 4 && pixel.r == 255) {
      pixel = {0, 0, 0};
      edgeQueue.push({x, y});
    }
  }

  for (int y = 0; y < in.height; y++) {
    for (int x = 0; x < in.width; x++) {
      int index = y * in.width + x;
      rgb8& pixel = in.buffer[index];

      if (pixel.r == 127 && pixel.g == 127 && pixel.b == 127) {
        pixel = {0, 0, 0};
      }
    }
  }

  return in;
}

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

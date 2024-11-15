#ifndef FILTER_HPP
#define FILTER_HPP


#include "Image.hpp"  // Assurez-vous que vous avez un fichier image.hpp avec la structure ImageView et rgb8
#include <vector>

// Appliquer une ouverture morphologique (érosion suivie de dilatation)
void morphologicalOpening(ImageView<rgb8> image, int radius);
ImageView<rgb8> HysteresisThreshold(ImageView<rgb8> in, int lowThreshold, int highThreshold);

#endif // COLOR_UTILS_HPP

#ifndef FILTER_HPP
#define FILTER_HPP


#include "Image.hpp"  // Assurez-vous que vous avez un fichier image.hpp avec la structure ImageView et rgb8


// Appliquer une ouverture morphologique (érosion suivie de dilatation)
void morphologicalOpening(ImageView<rgb8>& image, int radius);
void hysteresis(ImageView<rgb8>& image, int lowThreshold, int highThreshold);

#endif // COLOR_UTILS_HPP

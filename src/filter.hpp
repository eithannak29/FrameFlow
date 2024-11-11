#ifndef FILTER_HPP
#define FILTER_HPP


#include "Image.hpp"  // Assurez-vous que vous avez un fichier image.hpp avec la structure ImageView et rgb8


// Appliquer une ouverture morphologique (érosion suivie de dilatation)
void morphologicalOpening(ImageView<rgb8>& image, int radius);
ImageView<rgb8> HysteresisThreshold(ImageView<rgb8> in, int lowThreshold, int highThreshold);
ImageView<rgb8> applyRedMask(ImageView<rgb8> in, const ImageView<rgb8>& mask, std::vector<rgb8> initialPixels);

#endif // COLOR_UTILS_HPP

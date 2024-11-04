#pragma once

#include "image.hpp"  // Assurez-vous que vous avez un fichier image.hpp avec la structure ImageView et rgb8


// Appliquer une ouverture morphologique (érosion suivie de dilatation)
void morphologicalOpening(ImageView<rgb8>& image, int radius);


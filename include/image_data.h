#ifndef __IMAGE_DATA_H__
#define __IMAGE_DATA_H__

#include "cstdint"
#include "math.h"

// Maximum image width in pixels
constexpr uint16_t MAX_WIDTH = 1920;

// Maximum image height in pixels
constexpr uint16_t MAX_HEIGHT = 1080;

// Maximum number of color channels in the image
constexpr uint8_t MAX_CHANNELS = 3;

/**
 * @struct Image
 * @brief Struct to encapsulate image data and metadata for transport.
 *
 * The Image struct is designed to store the raw pixel data for an image along
 * with associated metadata such as the image's dimensions, the number of color
 * channels, and a timestamp. The timestamp can be used for synchronization or
 * ordering in systems where timing is critical.
 */
struct Image
{
    uint64_t index = 0;
    uint16_t width = MAX_WIDTH;
    uint16_t height = MAX_HEIGHT;
    uint8_t channels = MAX_CHANNELS;
    double_t timestamp = 0.;
    uint8_t image[MAX_WIDTH * MAX_HEIGHT * MAX_CHANNELS];
};

#endif // __IMAGE_DATA_H__
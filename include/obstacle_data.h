#ifndef __OBSTACLE_DATA_H__
#define __OBSTACLE_DATA_H__

#include "cstdint"
#include "math.h"

// Maximum obstacle
constexpr uint16_t MAX_OBS_NUM = 16;

/**
 * @struct Obstacle
 * @brief Struct to one Obstacle data for transport.
 */
struct Obstacle
{
    int cx;
    int cy;
    int w;
    int h;
    float conf;
    int classid;
};

/**
 * @struct Obstacles
 * @brief Struct to encapsulate Obstacles data and metadata for transport.
 */
struct Obstacles
{
    uint8_t obs_num = 0;               // Obstacles number
    double_t timestamp = 0.;           // timestamp
    Obstacle obstacles[MAX_OBS_NUM];   // Obstacles ptr
};

#endif // __OBSTACLE_DATA_H__

#ifndef __PRED_DATA_H__
#define __PRED_DATA_H__

#include "cstdint"
#include "math.h"
constexpr uint16_t MAX_OBS_NUM = 16;


struct Obstacle
{
    int cx;
    int cy;
    int w;
    int h;
    float conf;
    int classid;
};

struct Obstacles
{
    uint8_t obs_num = 0;
    double_t timestamp = 0.;
    Obstacle obstacles[MAX_OBS_NUM];
};

#endif // __PRED_DATA_H__

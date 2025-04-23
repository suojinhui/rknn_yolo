#ifndef __UTILS_H__
#define __UTILS_H__

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "RgaUtils.h"

#include "rknn_api.h"
#include <string>
#include "spdlog/spdlog.h"

const int RK3588 = 3;


/// Set the core of the model that needs to be bound
static int get_core_num()
{
    static int core_num = 0;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);

    int temp = core_num % RK3588;
    core_num++;
    return temp;
}

/// Anchor dimensions for YOLO detection layer 0, 1, 2 (stride 8, 16, 32)
const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};


/**
 * @brief Print tensor attributes for debugging
 * @param attr Pointer to rknn_tensor_attr structure
 * @details Logs detailed tensor information including:
 * - Dimensions and shape
 * - Data format and type
 * - Quantization parameters
 */
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

  SPDLOG_INFO("  index={}, name={}, n_dims={}, dims=[{}], n_elems={}, size={}, w_stride = {}, size_with_stride={}, fmt={}, type={}, qnt_type={}, zp={}, scale={}",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

/**
 * @brief Calculate intersection-over-union (IoU) of two rectangles
 * @param xmin0 Left coordinate of first rectangle
 * @param ymin0 Top coordinate of first rectangle
 * @param xmax0 Right coordinate of first rectangle
 * @param ymax0 Bottom coordinate of first rectangle
 * @param xmin1 Left coordinate of second rectangle
 * @param ymin1 Top coordinate of second rectangle
 * @param xmax1 Right coordinate of second rectangle
 * @param ymax1 Bottom coordinate of second rectangle
 * @return IoU value between 0.0 and 1.0
 */
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}


/**
 * @brief Perform class-aware Non-Maximum Suppression (NMS)
 * @param validCount Number of valid detections
 * @param outputLocations Vector of bounding box coordinates [x,y,w,h]
 * @param classIds Vector of class IDs for each detection
 * @param order Vector of sorted detection indices
 * @param filterId Class ID to perform NMS for
 * @param threshold IoU threshold for suppression
 * @return Always returns 0
 * @note Modifies order vector by setting suppressed indices to -1
 */
static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

/**
 * @brief Perform in-place descending quick sort on input values
 * @param input Vector of values to sort
 * @param left Left index of sort range
 * @param right Right index of sort range
 * @param indices Vector of indices to track original positions
 * @return Pivot index after partition
 * @note Modifies both input and indices vectors
 */
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

/**
 * @brief Calculate sigmoid function
 * @param x Input value
 * @return Sigmoid output in range (0,1)
 */
static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

/**
 * @brief Calculate inverse sigmoid (logit) function
 * @param y Value in range (0,1)
 * @return Logit output value
 */
static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

/**
 * @brief Clip value to specified range
 * @param val Input value
 * @param min Minimum allowed value
 * @param max Maximum allowed value
 * @return Clamped value between min and max
 */
inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

/**
 * @brief Quantize float32 value to affine quantized int8
 * @param f32 Input float value
 * @param zp Zero point for quantization
 * @param scale Scale factor for quantization
 * @return Quantized int8 value
 */
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

/**
 * @brief Dequantize affine quantized int8 to float32
 * @param qnt Quantized int8 value
 * @param zp Zero point used in quantization
 * @param scale Scale factor used in quantization
 * @return Dequantized float32 value
 */
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }


/**
 * @brief Process YOLO detection layer output
 * @param input Quantized output tensor pointer
 * @param anchor Anchor dimensions for this layer
 * @param grid_h Number of grid cells vertically
 * @param grid_w Number of grid cells horizontally
 * @param height Original model input height
 * @param width Original model input width
 * @param stride Detection layer stride
 * @param boxes Output vector for detected bounding boxes
 * @param objProbs Output vector for objectness probabilities
 * @param classId Output vector for class IDs
 * @param threshold Confidence threshold
 * @param zp Zero point for tensor dequantization
 * @param scale Scale factor for tensor dequantization
 * @param num_classes Number of detection classes
 * @return Number of valid detections
 * @details Performs:
 * - Anchor box decoding
 * - Confidence threshold filtering
 * - Class probability calculation
 * - Box coordinate conversion
 */
static int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride, 
                    std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                    int32_t zp, float scale, int num_classes)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int8_t box_confidence = input[((num_classes + 5) * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8)
                {
                    int offset = ((num_classes + 5) * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < num_classes; ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8)
                    {
                        objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

/**
 * @brief Clamp value between min and max
 * @param val Input value
 * @param min Minimum allowed value
 * @param max Maximum allowed value
 * @return Clamped integer value
 */
inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

/**
 * @brief Convert timeval struct to microseconds
 * @param t timeval structure
 * @return Time in microseconds
 */
static double _get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

#endif //__UTILS_H__
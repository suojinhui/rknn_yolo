#ifndef __MODEL_H__
#define __MODEL_H__

#include <string>
#include <opencv2/opencv.hpp>
// rga
#include "RgaUtils.h"
#include "im2d.h"
//rknn
#include "rknn_api.h"

#include "label.h"
#include "pred_data.h"

/// Maximum number of detected objects
#define OBJ_NUMB_MAX_SIZE 64

/**
 * @brief Structure representing basic image information
 * @author suojinhui
 */
struct image_info {
    int h;      ///< Image height in pixels
    int w;      ///< Image width in pixels 
    int c;      ///< Number of color channels
    image_info(int height, int width, int channel) : h(height), w(width), c(channel) {}
    image_info() : h(0), w(0), c(0) {}
};

/**
 * @brief Structure representing bounding box coordinates
 */
typedef struct _BOX_RECT {
    int left;   ///< Left x-coordinate
    int right;  ///< Right x-coordinate
    int top;    ///< Top y-coordinate
    int bottom; ///< Bottom y-coordinate
} BOX_RECT;

/**
 * @brief Structure representing single detection result
 */
typedef struct _detect_result_t {
    int id;         ///< Class ID
    BOX_RECT box;   ///< Bounding box coordinates
    float prop;     ///< Confidence score [0,1]
} detect_result_t;

/**
 * @brief Structure representing group of detection results
 */
typedef struct _detect_result_group_t {
    int id;                         ///< Result group identifier
    int count;                      ///< Number of valid detections
    detect_result_t results[OBJ_NUMB_MAX_SIZE]; ///< Array of detection results
} detect_result_group_t;

/**
 * @brief YOLO model processing class for RKNN platform
 * 
 * Encapsulates the complete inference pipeline including model loading, 
 * preprocessing, NPU acceleration, and postprocessing for YOLO models.
 */
class YOLO_MODEL {
public:
    /**
     * @brief Construct a new YOLO model processor
     * @param model_name Path to RKNN model file
     * @param num_classes Number of detection classes
     * @param nms_threshold Threshold for Non-Maximum Suppression [0,1]
     * @param box_conf_threshold Minimum confidence score for detection [0,1]
     * @param input_width Model input width in pixels
     * @param input_height Model input height in pixels
     * @param dataset_type Type of dataset (COCO/Bdd100k) for label mapping
     */
    YOLO_MODEL(const char* model_name, int num_classes, float nms_threshold, float box_conf_threshold, 
        int input_width, int input_height, Datasets dataset_type);
    
    /**
     * @brief Default destructor
     * @note Actually, this function will not perform a thorough analysis. Please call the destroy function
     */
    ~YOLO_MODEL();

    // Runtime functions
    
    /**
     * @brief Execute complete inference pipeline
     * @param orig_img Input image in BGR format (OpenCV default)
     * @return cv::Mat with detected box
     * @note Modifies input image to draw detection results
     */
    bool inference(const cv::Mat& orig_img, Obstacles& obs, const double_t& timestamp);

    /**
     * @brief Preprocess input image for model inference
     * @param orig_img Input image in BGR format
     * @return true if preprocessing succeeded
     * @return false if preprocessing failed
     * @details Performs:
     * - Color space conversion (BGR→RGB)
     * - Image resizing using RGA hardware
     */
    bool preprocess(const cv::Mat& orig_img);

    /**
     * @brief Postprocess model outputs to detection results
     * @details Performs:
     * - Output tensor dequantization
     * - Anchor decoding for three detection layers
     * - Confidence threshold filtering
     * - Class-aware Non-Maximum Suppression
     * - Coordinate scaling to original model input size, not original image input size!
     */
    void postprocess();

    /**
     * @brief Reset detection results buffer
     * @note Should be called before new inference
     */
    void reset_task();

    /**
     * @brief Release all allocated resources
     * @details Releases:
     * - RKNN context and memory buffers
     * - RGA buffer handles
     * - Internal image buffers
     */
    void destroy();

    // Initialization functions
    
    /**
     * @brief Initialize complete model pipeline
     * @note Must be called before first inference
     * @throws Terminates application on critical errors
     */
    void init_model();

    /**
     * @brief Load RKNN model and query model attributes
     * @return true if model loaded successfully
     * @return false if loading failed
     * @details Performs:
     * - Model file loading
     * - RKNN context creation
     * - Model version/IO/attribute queries
     */
    bool load_model();

    /**
     * @brief Initialize model input/output buffers
     * @return true if buffer initialization succeeded
     * @return false if initialization failed
     * @note Uses RKNN API to allocate NPU-side memory
     */
    bool init_io_buffer();

    /**
     * @brief Initialize RGA buffers for hardware acceleration
     * @return true if RGA initialized successfully
     * @return false if RGA initialization failed
     * @details Sets up:
     * - Source buffer for original image
     * - Destination buffer aligned with model input
     */
    bool init_rga_buffer();

public:
    // RKNN components
    rknn_context model_rk_context_;     ///< RKNN execution context
    rknn_sdk_version version_;          ///< RKNN SDK version info
    rknn_input_output_num io_num_;      ///< Model I/O specifications
    rknn_tensor_attr* input_attrs_;     ///< Input tensor attributes
    rknn_tensor_attr* output_attrs_;    ///< Output tensor attributes

    rknn_tensor_mem* model_inputs_[1];  ///< Input memory buffers
    rknn_tensor_mem* model_outputs_[3]; ///< Output memory buffers
    rknn_tensor_format input_format_;   ///< Model input data format

    // Model configuration
    int num_classes_;                   ///< Number of detection classes
    image_info image_info_;             ///< Model input specifications
    image_info image_info_input_;       ///< Actual input image specs
    float nms_threshold_;               ///< NMS overlap threshold
    float box_conf_threshold_;          ///< Detection confidence threshold

    // Model file info
    const char* model_name_;            ///< Path to RKNN model file

    // RGA components
    rga_buffer_t src_;                  ///< Source image buffer
    rga_buffer_t dst_;                  ///< Destination image buffer
    rga_buffer_handle_t src_handle_;    ///< Source buffer handle
    rga_buffer_handle_t dst_handle_;    ///< Destination buffer handle

    uint8_t* image_;                    ///< Preprocessed image buffer

    // Label management
    std::shared_ptr<IBaseLabels> labels_map_; ///< Dataset label mapper
    detect_result_group_t detect_result_group_; ///< Detection results buffer
    std::vector<float> out_scales_;      ///< Output tensor scaling factors
    std::vector<int32_t> out_zps_;       ///< Output tensor zero points
};

/**
 * @brief Factory function to create YOLO detector instance
 * @param model_name Path to RKNN model file
 * @param num_classes Number of detection classes
 * @param nms_threshold NMS overlap threshold [0,1]
 * @param box_conf_threshold Detection confidence threshold [0,1]
 * @param input_width Model input width in pixels
 * @param input_height Model input height in pixels
 * @param dataset_type Dataset type for label mapping
 * @return std::shared_ptr<YOLO_MODEL> Shared pointer to detector instance
 */
std::shared_ptr<YOLO_MODEL> make_detector(const char* model_name, int num_classes, float nms_threshold, 
    float box_conf_threshold, int input_width, int input_height, Datasets dataset_type);

#endif // __MODEL_H__
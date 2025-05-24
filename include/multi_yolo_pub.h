#ifndef __MULTI_YOLO_PUB_H__
#define __MULTI_YOLO_PUB_H__ 

#include "iostream"
#include "condition_variable"
#include "opencv2/opencv.hpp"
#include <opencv2/core/ocl.hpp>
#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "obstacle_data.h"
#include "yolo_node.h"
#include "obs_publisher.h"

/**
 * @class Multi_yolo_pub
 * @brief Multi-node YOLO obstacle detection and publishing system
 * 
 * Orchestrates multiple YOLO detection pipelines with features:
 * - Parallel image processing from multiple sources
 * - Iceoryx-based inter-process communication
 * - Dynamic configuration loading
 * - Graceful shutdown handling
 * - Integrated logging system
 * 
 * @note Each detection pipeline contains:
 *       1. Image subscriber
 *       2. YOLO detection model
 *       3. Obstacle publisher
 */
class Multi_yolo_pub
{
private:
    std::string config_path_; // Path to YAML configuration file

    // Iceoryx communication settings
    std::string iceoryx_runtime_name_ = "RK_YOLO"; // Iceoryx communication settings

    std::string iceoryx_service_ = "camera"; // Service name for image subscribers
    std::string image_instance_ = "image"; // Instance identifier for image channels
    std::vector<std::string> image_topics_ = {}; // List of image topic names to subscribe

    std::string iceoryx_obs_service_ = "yolo"; // Service name for obstacle publishers
    std::string obs_instance_ = "obstacle"; // Instance identifier for obstacle data
    std::vector<std::string> obs_topics_ = {}; // List of obstacle topics to publish

    // YOLO model configuration
    std::string model_path_; // Path to ONNX model file
    int num_classes_; // Number of detection classes
    float nms_threshold_; // Non-Maximum Suppression threshold
    float box_conf_threshold_; // Bounding box confidence threshold
    int input_width_; // Model input tensor width
    int input_height_; // Model input tensor height

    // Logging configuration
    std::string log_dir_ = "../log"; // Directory path for log storage
    std::string log_level_ = "info"; // Log severity level (info/debug/etc)
    int log_retention_days_ = 3; // Days to retain log files

public:
    /**
     * @brief Construct a new Muti_yolo_pub instance
     * @param config_path Path to YAML configuration file
     * 
     * @note Actual initialization happens in Start()
     */
    Multi_yolo_pub(const std::string &config_path);
    
    /// @brief Destructor cleans up system resources
    ~Multi_yolo_pub();

    /**
     * @brief Load configurations from YAML file
     * @return true if configuration loaded successfully
     * @return false if file error or invalid parameters
     * 
     * @throws cv::Exception on malformed YAML content
     */
    bool LoadConfigs();

    /**
     * @brief Start the multi-node detection system
     * 
     * Orchestrates:
     * 1. Configuration loading
     * 2. Logger initialization
     * 3. Iceoryx runtime setup
     * 4. Parallel node initialization
     * 5. Signal handling
     * 6. Clean shutdown sequence
     */
    void Start();

private:
    /**
     * @brief Publish obstacles from detection node
     * @param yolo_node Shared pointer to detection node
     * @param publisher Shared pointer to obstacle publisher
     * 
     * @details Runs in dedicated thread until node stops:
     * - Continuously fetches latest detection results
     * - Publishes via Iceoryx shared memory
     */
    void PublishObs(std::shared_ptr<Yolo_node> yolo_node,
                    std::shared_ptr<ObstaclePublisher> publisher);

};


#endif // __MULTI_YOLO_PUB_H__
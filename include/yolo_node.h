#ifndef __YOLO_NODE_H__
#define __YOLO_NODE_H__

#include "iostream"
#include "condition_variable"
#include "opencv2/opencv.hpp"
#include <opencv2/core/ocl.hpp>
#include "image_subscriber.h"
#include "obstacle_data.h"
#include "image_subscriber.h"
#include "model.h"

/**
 * @class Yolo_node
 * @brief YOLO detection processing node with thread synchronization
 * 
 * Manages the complete detection pipeline lifecycle:
 * - Image acquisition from subscriber
 * - YOLO model inference
 * - Obstacle data synchronization
 * - Thread-safe result access
 */
class Yolo_node
{
private:
    bool is_detect_ready_; // Detection result availability flag
    std::mutex detect_ready_mutex_; // Guards access to detection status
    std::condition_variable detect_ready_condition_; // Notifies consumers of new detection results

    std::string name_; // Diagnostic identifier for multi-node systems

    bool is_stopped_ = false; // Atomic termination flag for detection thread
    std::mutex stopped_mutex_; // Synchronizes stop state modifications

    Obstacles obstacles_; // Latest detected obstacle data container

    std::thread detect_thread_; // Worker thread for continuous detection

    /**
     * @brief Core detection processing loop
     * @param subscriber Shared image source
     * @param detector YOLO model instance
     * 
     * @details Continuous workflow:
     * 1. Acquires image with timestamp
     * 2. Performs model inference
     * 3. Updates obstacle data
     * 4. Notifies result consumers
     */
    void Detect_form_subscriber(std::shared_ptr<ImageSubscriber> subscriber, std::shared_ptr<YOLO_MODEL> detector);

public:
    /**
     * @brief Construct a new Yolo_node instance
     * @param name Unique identifier for logging/diagnostics
     */
    Yolo_node(std::string name);
    ~Yolo_node();

    /**
     * @brief Launches detection thread
     * @param subscriber Image source to attach
     * @param detector Model instance to utilize
     */
    void Start_detect_loop(std::shared_ptr<ImageSubscriber> subscriber, std::shared_ptr<YOLO_MODEL> detector);

    /**
     * @brief Retrieves latest detection results
     * @param[out] obstacles Reference to populate with results
     * @return true if valid data retrieved, false on termination
     * 
     * @note Blocks until new detection or stop signal
     */
    bool Get_obstacles(Obstacles& obstacles);

    /**
     * @brief Initiates graceful thread termination
     * 
     * 1. Sets stop flag
     * 2. Joins worker thread
     * 3. Releases resources
     */
    void Stop();

    /**
     * @brief Checks thread termination status
     * @return true if stop sequence initiated
     */
    bool IsStopped();

};

#endif // __YOLO_NODE_H__
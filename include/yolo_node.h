#ifndef __YOLO_NODE_H__
#define __YOLO_NODE_H__

#include "iostream"
#include "condition_variable"
#include "opencv2/opencv.hpp"
#include <opencv2/core/ocl.hpp>
#include "image_subscriber.h"
#include "pred_data.h"
#include "image_subscriber.h"
#include "model.h"

class Yolo_node
{
private:
    bool is_detect_ready_;
    std::mutex detect_ready_mutex_;
    std::condition_variable detect_ready_condition_;

    std::string name_;

    bool is_stopped_ = false;
    std::mutex stopped_mutex_;

    Obstacles obstacles_;

    std::thread detect_thread_;

public:
    Yolo_node(std::string name);
    ~Yolo_node();

    void detect_form_subscriber(std::shared_ptr<ImageSubscriber> subscriber, std::shared_ptr<YOLO_MODEL> detector);

    void start_detect_loop(std::shared_ptr<ImageSubscriber> subscriber, std::shared_ptr<YOLO_MODEL> detector);

    bool get_obstacles(Obstacles& obstacles);

    void stop();

    bool IsStopped();

};

#endif // __YOLO_NODE_H__
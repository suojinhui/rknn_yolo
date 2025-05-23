#ifndef __MUTI_YOLO_PUB_H__
#define __MUTI_YOLO_PUB_H__ 

#include "iostream"
#include "condition_variable"
#include "opencv2/opencv.hpp"
#include <opencv2/core/ocl.hpp>
#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "pred_data.h"
#include "yolo_node.h"
#include "obs_publisher.h"

class Muti_yolo_pub
{
private:
    std::string config_path_;

    std::string iceoryx_runtime_name_ = "RK_YOLO";

    std::string iceoryx_service_ = "camera";
    std::string image_instance_ = "image";
    std::vector<std::string> image_topics_ = {};

    std::string iceoryx_obs_service_ = "yolo";
    std::string obs_instance_ = "obstacle";
    std::vector<std::string> obs_topics_ = {};

    std::string model_path_;
    int num_classes_;
    float nms_threshold_;
    float box_conf_threshold_;
    int input_width_;
    int input_height_;

    std::string log_dir_ = "../log";
    std::string log_level_ = "info";
    int log_retention_days_ = 3;

public:
    Muti_yolo_pub(const std::string &config_path);
    ~Muti_yolo_pub();

    bool LoadConfigs();

    void Start();

private:
    void PublishObs(std::shared_ptr<Yolo_node> yolo_node, std::shared_ptr<ObstaclePublisher> publisher);

};


#endif // __MUTI_YOLO_PUB_H__
#include "yolo_node.h"

Yolo_node::Yolo_node(std::string name): name_(name) {}

Yolo_node::~Yolo_node()
{}

void Yolo_node::Detect_form_subscriber(std::shared_ptr<ImageSubscriber> subscriber, std::shared_ptr<YOLO_MODEL> detector){
    cv::Mat image;
    double_t timestamp;
    while (!is_stopped_)
    {
        if(subscriber->GetImageCopyWithExposureTimeStamp(image, timestamp)){
            if (detector->inference(image, obstacles_, timestamp))
            {
                std::unique_lock<std::mutex> lock(detect_ready_mutex_);
                is_detect_ready_ = true;
                detect_ready_condition_.notify_all();
            }
            else
            {
                std::cerr << name_ + ": Failed to inference image" << std::endl;
            }
        }
        else
        {
            std::cerr << name_ + ": Failed to subscribe image" << std::endl;
        }
    }

    return;

}

void Yolo_node::Start_detect_loop(std::shared_ptr<ImageSubscriber> subscriber, std::shared_ptr<YOLO_MODEL> detector){
    detect_thread_ = std::thread(&Yolo_node::Detect_form_subscriber, this, subscriber, detector);
    return;
}

bool Yolo_node::Get_obstacles(Obstacles& obstacles){
    {
        std::unique_lock<std::mutex> lock(detect_ready_mutex_);
        detect_ready_condition_.wait(lock, [this]
                                    { return is_detect_ready_ || is_stopped_; });
        if (is_stopped_)
        {
            return false;
        }
        is_detect_ready_ = false;
    }

    obstacles = obstacles_;
    return true;
}

void Yolo_node::Stop(){
    {
        std::unique_lock<std::mutex> lock(stopped_mutex_);
        is_stopped_ = true;
    }

    if (detect_thread_.joinable())
    {
        detect_thread_.join();
    }
    return;
}

bool Yolo_node::IsStopped()
{
    return is_stopped_;
}
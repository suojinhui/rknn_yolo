#include "multi_yolo_pub.h"
#include "utils.h"
#include "logger.h"
#include "signal_handler.h"
#include "image_subscriber.h"
#include "obs_publisher.h"
#include "model.h"
#include "yolo_node.h"
#include "label.h"

Multi_yolo_pub::Multi_yolo_pub(const std::string &config_path): config_path_(config_path)
{}

Multi_yolo_pub::~Multi_yolo_pub()
{}

bool Multi_yolo_pub::LoadConfigs() {
    cv::FileStorage configs(config_path_, cv::FileStorage::READ);
    if (!configs.isOpened()) {
        std::cerr << "Error: Unable to open the configuration file: " << config_path_ << std::endl;
        return false;
    }

    try {
        // helper lambda to check key existence and read
        auto readConfig = [&](const std::string &key, auto &var) -> bool {
            if (!configs[key].empty()) {
                configs[key] >> var;
                return true;
            } else {
                std::cerr << "Error: Missing key in config file: " << key << std::endl;
                return false;
            }
        };

        bool success = true;

        // iceoryx configs
        success &= readConfig("iceoryx_runtime_name", iceoryx_runtime_name_);

        // subscriber image configs
        success &= readConfig("iceoryx_service", iceoryx_service_);
        success &= readConfig("image_instance", image_instance_);
        success &= readConfig("image_topics", image_topics_);

        // publisher obstacle configs
        success &= readConfig("iceoryx_obs_service", iceoryx_obs_service_);
        success &= readConfig("obs_instance", obs_instance_);
        success &= readConfig("obs_topics", obs_topics_);

        // yolo model build config
        success &= readConfig("model_path", model_path_);
        success &= readConfig("num_classes", num_classes_);
        success &= readConfig("nms_threshold", nms_threshold_);
        success &= readConfig("box_conf_threshold", box_conf_threshold_);
        success &= readConfig("input_width", input_width_);
        success &= readConfig("input_height", input_height_);

        // log configs
        success &= readConfig("log_dir", log_dir_);
        success &= readConfig("log_level", log_level_);
        success &= readConfig("log_retention_days", log_retention_days_);

        configs.release();

        if (!success) {
            std::cerr << "Error: Configuration loading failed due to missing keys." << std::endl;
            std::abort();
            return false;
        }
    }
    catch (const cv::Exception &error) {
        std::cerr << "Error: Failed to read configuration parameters. Exception: " << error.what() << std::endl;
        return false;
    }

    return true;
}

void Multi_yolo_pub::Start(){

    // initiate Logger
    Logger::GetInstance(log_level_, log_dir_, log_retention_days_);

    // initiate SignalHandler
    SignalHandler::SetSignalAction();

    // initiate iceoryx runtime
    char iceoryx_runtime_name[80] = {0};
    strncpy(iceoryx_runtime_name, iceoryx_runtime_name_.c_str(), sizeof(iceoryx_runtime_name));
    try
    {
        iox::runtime::PoshRuntime::initRuntime(iceoryx_runtime_name);
    }
    catch (const std::exception &error)
    {
        std::cerr << "Error: Failed to initialize Iceoryx runtime. Exception: " << error.what() << std::endl;
        SPDLOG_ERROR("Error: Failed to read configuration parameters. Exception:{}", error.what());
        return;
    }

    int node_nums = image_topics_.size();
    SPDLOG_INFO("The number of enabled detection nodes is {}", node_nums);
    
    std::vector<std::shared_ptr<ImageSubscriber>> subscribers(node_nums);
    std::vector<std::shared_ptr<ObstaclePublisher>> publishers(node_nums);
    std::vector<std::shared_ptr<Yolo_node>> yolo_nodes(node_nums);
    std::vector<std::shared_ptr<YOLO_MODEL>> detectors(node_nums);

    std::vector<std::thread> publish_threads(node_nums);

#pragma omp parallel for
    for (int i = 0; i < node_nums; i++)
    {
        subscribers[i] = std::make_shared<ImageSubscriber>(iceoryx_service_, image_instance_, image_topics_[i]);
        publishers[i] = std::make_shared<ObstaclePublisher>(iceoryx_obs_service_, obs_instance_, obs_topics_[i]);
        yolo_nodes[i] = std::make_shared<Yolo_node>(image_topics_[i]);
        detectors[i] = std::make_shared<YOLO_MODEL>(model_path_.c_str(), num_classes_, nms_threshold_, box_conf_threshold_, 
                                                        input_width_, input_height_, Datasets::COCO);
        
        detectors[i]->init_model();

        yolo_nodes[i]->Start_detect_loop(subscribers[i], detectors[i]);

        publish_threads[i] = std::thread(&Multi_yolo_pub::PublishObs, this, yolo_nodes[i], publishers[i]);
    }

    SignalHandler::GetInstance().WaitForStopSignal();

#pragma omp parallel for
    for (int i = 0; i < node_nums; i++)
    {
        yolo_nodes[i]->Stop();
        subscribers[i]->Stop();
        detectors[i]->destroy();
    }

#pragma omp parallel for
    for (int i = 0; i < node_nums; ++i)
    {
        if (publish_threads[i].joinable())
        {
            publish_threads[i].join();
        }
    }

    iox::runtime::PoshRuntime::getInstance().shutdown();

    Logger::GetInstance().Stop();
    return;
}

void Multi_yolo_pub::PublishObs(std::shared_ptr<Yolo_node> yolo_node, std::shared_ptr<ObstaclePublisher> publisher){

    Obstacles obs;

    while (!yolo_node->IsStopped())
    {
        if (yolo_node->Get_obstacles(obs))
        {
            publisher->PublishObstacle(obs);
        }
    }
    return;
}
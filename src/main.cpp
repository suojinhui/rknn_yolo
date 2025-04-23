# include <iostream>
# include <opencv2/opencv.hpp>

#include "model.h"
#include "label.h"
#include "capturer.h"
#include "utils.h"

#include <signal.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <omp.h>
#include "rknnpool.h"

std::atomic<bool> quit(false);

void signal_handler(int signal) {
    if (signal == SIGINT) {
        quit.store(true);
    }
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);

    // set log level
    auto logger = spdlog::default_logger();
    logger->set_level(spdlog::level::debug);

    std::string model_path = "model/yolov5s-640-640.rknn";

    // create model pool
    int threadNum = 3; // rk3588 has 3 core npu
    rknnPool<YOLO_MODEL, cv::Mat, cv::Mat> modelPool(model_path.c_str(), 80, 0.45, 0.25, 1920, 1080, Datasets::COCO, threadNum);
    if (modelPool.init() != 0)
    {
        SPDLOG_INFO("rknnPool init fail!");
        return -1;
    }

    constexpr size_t NUM_CAMERAS = 1;
    std::vector<std::shared_ptr<capturer::Capturer>> capturers(NUM_CAMERAS);
    const std::vector<int> device_ids = {11};
    const std::string label = "Camera";

    struct timeval start_time, stop_time;
    
    //create camera
#pragma omp parallel for
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {
        capturers[i] = std::make_shared<capturer::Capturer>(
            device_ids[i], 1920, 1080, 1920, 1080, 30, 0, 
            label + "_" + std::to_string(i)
        );
        capturers[i]->StartCaptureThread();
    }

    std::vector<std::string> window_names;
    for (size_t i = 0; i < 3; ++i) {
        std::string win_name = label + "_" + std::to_string(i);
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(win_name, 640, 360);
        window_names.push_back(win_name);
    }

    while (!quit.load()) {
        std::vector<cv::Mat> frames(NUM_CAMERAS);
        
#pragma omp parallel for
        for (size_t i = 0; i < NUM_CAMERAS; ++i) {
            frames[i] = capturers[i]->GetImage().clone();
        }

        for (size_t i = 0; i < NUM_CAMERAS; ++i) {
            if (!frames[i].empty()) {
                gettimeofday(&start_time, NULL);
                modelPool.put(frames[i].clone());
                modelPool.put(frames[i].clone());
                modelPool.put(frames[i].clone());
                if (modelPool.get(frames[i]) == 0) {
                    cv::imshow(window_names[0], frames[i]);
                }
                if (modelPool.get(frames[i]) == 0) {
                    cv::imshow(window_names[1], frames[i]);
                }
                if (modelPool.get(frames[i]) == 0) {
                    cv::imshow(window_names[2], frames[i]);
                }
                gettimeofday(&stop_time, NULL);
                SPDLOG_INFO("run once: {} ms", (_get_us(stop_time) - _get_us(start_time)) / 2000);
            }
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') {
            quit.store(true);
        }
    }

    for (auto& capturer : capturers) {
        if (capturer) {
            capturer->StopCapture();
        }
    }

    for (const auto& win_name : window_names) {
        cv::destroyWindow(win_name);
    }

    return 0;
}
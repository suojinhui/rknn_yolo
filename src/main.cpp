#include "iostream"
#include "multi_yolo_pub.h"

int main(int argc, const char** argv) {
    std::cout << "Hello, this is Multi-node YOLO obstacle detection and publishing system!" << std::endl;

    Multi_yolo_pub multi_yolo_pub("./config/configs.yaml");
    multi_yolo_pub.LoadConfigs();
    multi_yolo_pub.Start();

    return 0;
}
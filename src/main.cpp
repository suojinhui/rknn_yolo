#include "iostream"
#include "muti_yolo_pub.h"

int main(int argc, const char** argv) {
    std::cout << "Hello, this is YOLO detectors for Electric Locomotive Telecontrol System!" << std::endl;

    Muti_yolo_pub muti_yolo_pub("./config/configs.yaml");
    muti_yolo_pub.Start();

    return 0;
}
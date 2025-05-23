#ifndef __IMAGE_SUBSCRIBER_H__
#define __IMAGE_SUBSCRIBER_H__

#include "mutex"
#include "condition_variable"

#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iceoryx_posh/popo/listener.hpp"
#include "iceoryx_posh/popo/subscriber.hpp"
#include "iox/optional.hpp"
#include "opencv2/opencv.hpp"

#include "image_data.h"
#include "topic_info.h"

class ImageSubscriber
{
private:
    TopicInfo topic_info_;
    std::string topic_;
    iox::popo::Subscriber<Image> subscriber_;
    iox::popo::Listener listener_;

    double_t timestamp_;

    cv::Mat image_;
    std::mutex image_ready_mutex_;
    std::condition_variable image_ready_condition_;

    static void OnDataAvailable(iox::popo::Subscriber<Image> *subscriber, ImageSubscriber *self);

public:
    ImageSubscriber(const std::string &service, const std::string &instance, const std::string &topic);

    ~ImageSubscriber();

    cv::Mat GetImage();

    bool GetImage(cv::Mat &image);

    bool GetImageWithExposureTimestamp(cv::Mat &image, double_t &timestamp);

    bool GetImageCopy(cv::Mat &image);

    bool GetImageCopyWithExposureTimeStamp(cv::Mat &image, double_t &timestamp);

    void Stop();
};

#endif // 
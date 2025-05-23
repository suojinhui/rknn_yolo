#include "iostream"

#include "image_subscriber.h"

void ImageSubscriber::OnDataAvailable(iox::popo::Subscriber<Image> *subscriber,
                                      ImageSubscriber *self)
{
    subscriber->take().and_then([subscriber, self](auto &sample)
                                {
                                    switch (sample->channels)
                                    {
                                    case 1:
                                        self->image_ = cv::Mat(sample->height, sample->width, CV_8UC1, const_cast<uint8_t *>(sample->image));
                                        break;
                                    case 2:
                                        self->image_ = cv::Mat(sample->height, sample->width, CV_8UC2, const_cast<uint8_t *>(sample->image));
                                        break;
                                    case 3:
                                        self->image_ = cv::Mat(sample->height, sample->width, CV_8UC3, const_cast<uint8_t *>(sample->image));
                                        break;
                                    default:
                                        self->image_ = cv::Mat(sample->height, sample->width, CV_8UC3, const_cast<uint8_t *>(sample->image));
                                        break;
                                    }

                                    self->timestamp_ = sample->timestamp;
                                    {
                                        std::unique_lock<std::mutex> lock(self->image_ready_mutex_);
                                        self->image_ready_condition_.notify_all();
                                    } });
}

ImageSubscriber::ImageSubscriber(const std::string &service,
                                 const std::string &instance,
                                 const std::string &topic)
    : topic_info_(TopicInfo(service, instance, topic)),
      subscriber_({topic_info_.service_, topic_info_.instance_, topic_info_.topic_}, topic_info_.subscriber_options_),
      topic_(service + '/' + instance + '/' + topic)

{
    listener_.attachEvent(subscriber_,
                          iox::popo::SubscriberEvent::DATA_RECEIVED,
                          iox::popo::createNotificationCallback(OnDataAvailable, *this))
        .or_else([](auto)
                 {
                std::cerr << "unable to attach subscriberLeft" << std::endl;
                std::exit(EXIT_FAILURE); });
}

ImageSubscriber::~ImageSubscriber() {}

cv::Mat ImageSubscriber::GetImage()
{
    std::unique_lock<std::mutex> lock(image_ready_mutex_);
    if (image_ready_condition_.wait_for(lock, std::chrono::milliseconds(60)) == std::cv_status::timeout)
    {
        return cv::Mat();
    }
    return image_;
}

bool ImageSubscriber::GetImage(cv::Mat &image)
{
    cv::Mat subscribered_image = GetImage();
    if (subscribered_image.empty())
    {
        return false;
    }
    image = subscribered_image;
    return true;
}

bool ImageSubscriber::GetImageWithExposureTimestamp(cv::Mat &image,
                                                    double_t &timestamp)
{
    cv::Mat subscribered_image = GetImage();
    if (subscribered_image.empty())
    {
        return false;
    }
    image = subscribered_image;
    timestamp = timestamp_;
    return true;
}

bool ImageSubscriber::GetImageCopy(cv::Mat &image)
{
    cv::Mat subscribered_image = GetImage();
    if (subscribered_image.empty())
    {
        return false;
    }
    subscribered_image.copyTo(image);
    return true;
}

bool ImageSubscriber::GetImageCopyWithExposureTimeStamp(cv::Mat &image,
                                                        double_t &timestamp)
{
    if (false == GetImageCopy(image))
    {
        return false;
    }
    timestamp = timestamp_;
    return true;
}

void ImageSubscriber::Stop()
{
    listener_.detachEvent(subscriber_, iox::popo::SubscriberEvent::DATA_RECEIVED);

    {
        std::unique_lock<std::mutex> lock(image_ready_mutex_);
        image_ready_condition_.notify_all();
    }
    return;
}
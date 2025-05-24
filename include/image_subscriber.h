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


/**
 * @class ImageSubscriber
 * @brief Iceoryx-based subscriber for receiving image data with OpenCV integration
 *
 * Provides thread-safe image acquisition through shared memory communication.
 * Supports multiple image formats (mono/color) and exposure timestamp tracking.
 */
class ImageSubscriber
{
private:
    TopicInfo topic_info_; // service/instance/topic struct data
    std::string topic_; // service/instance/topic
    iox::popo::Subscriber<Image> subscriber_; // Iceoryx Subscriber
    iox::popo::Listener listener_; // Event listener, used to register data arrival callback

    double_t timestamp_;
    cv::Mat image_;

    std::mutex image_ready_mutex_;
    std::condition_variable image_ready_condition_;

    /**
     * @brief Static callback for handling incoming image data notifications
     * 
     * @param subscriber Pointer to the Iceoryx subscriber that triggered the event
     * @param self Pointer to the owning ImageSubscriber instance context
     */
    static void OnDataAvailable(iox::popo::Subscriber<Image> *subscriber, ImageSubscriber *self);

public:

    /**
     * @brief Construct a new ImageSubscriber with service discovery parameters
     * @param service Service identifier for iceoryx communication
     * @param instance Instance identifier for service grouping
     * @param topic Specific data channel within the instance
     */
    ImageSubscriber(const std::string &service, const std::string &instance, const std::string &topic);

    ~ImageSubscriber();

    /**
     * @brief Blocking call to retrieve latest image
     * @return cv::Mat containing received image data (empty on timeout)
     */
    cv::Mat GetImage();

    /**
     * @brief Non-blocking image retrieval with status feedback
     * @param[out] image Reference to store received image
     * @return true if valid image received, false otherwise
     */
    bool GetImage(cv::Mat &image);

    /**
     * @brief Retrieve image with associated exposure timestamp
     * @param[out] image Reference to store received image
     * @param[out] timestamp Exposure timestamp in double precision
     * @return true if valid data received, false otherwise
     */
    bool GetImageWithExposureTimestamp(cv::Mat &image, double_t &timestamp);

    /**
     * @brief Get a deep copy of the current image
     * @param[out] image Reference to store image copy
     * @return true if valid copy created, false otherwise
     */
    bool GetImageCopy(cv::Mat &image);

    /**
     * @brief Get image copy with associated exposure timestamp
     * @param[out] image Reference to store image copy
     * @param[out] timestamp Exposure timestamp in double precision
     * @return true if valid data received, false otherwise
     */
    bool GetImageCopyWithExposureTimeStamp(cv::Mat &image, double_t &timestamp);

    /**
     * @brief Gracefully stop subscription and release resources
     */
    void Stop();
};

#endif // __IMAGE_SUBSCRIBER_H__
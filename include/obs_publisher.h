#ifndef __OBS_PUBLISHER_H__
#define __OBS_PUBLISHER_H__

#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iceoryx_posh/popo/publisher.hpp"
#include "opencv2/opencv.hpp"

#include "topic_info.h"
#include "obstacle_data.h"


/**
 * @class ObstaclePublisher
 * @brief Iceoryx-based publisher for obstacle detection results
 * 
 * Provides zero-copy publishing of obstacle data through shared memory.
 * Implements thread-safe data transmission with configurable QoS.
 */
class ObstaclePublisher
{
private:
    TopicInfo topic_info_; // Topic metadata container (service/instance/topic triad)
    std::string topic_; // Full topic URI in "service/instance/topic" format
    iox::popo::Publisher<Obstacles> publisher_; // Iceoryx publisher instance for obstacle data
    unsigned long index_ = 0; // Sequence counter for debugging/monitoring (currently unused)

public:
    /**
     * @brief Construct a new ObstaclePublisher instance
     * @param service Service identifier for DDS-like discovery
     * @param instance Logical group within the service
     * @param topic Specific data channel name
     * 
     * @note Automatically registers with Iceoryx runtime
     */
    ObstaclePublisher(const std::string &service, const std::string &instance, const std::string &topic);
    ~ObstaclePublisher();

    /**
     * @brief Publish obstacle detection results
     * @param obs Obstacles data structure to transmit
     * 
     * @details This method:
     * 1. Acquires shared memory segment
     * 2. Copies obstacle data with timestamp
     * 3. Atomically publishes the sample
     * 
     * @warning May block if shared memory is exhausted
     * @note Logs transmission status via SPDLOG
     */

    void PublishObstacle(const Obstacles &obs);
};

#endif // __OBS_PUBLISHER_H__
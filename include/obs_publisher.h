#ifndef __OBS_PUBLISHER_H__
#define __OBS_PUBLISHER_H__

#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iceoryx_posh/popo/publisher.hpp"
#include "opencv2/opencv.hpp"

#include "topic_info.h"
#include "pred_data.h"

class ObstaclePublisher
{
private:
    TopicInfo topic_info_;
    std::string topic_;
    iox::popo::Publisher<Obstacles> publisher_;
    unsigned long index_ = 0;

public:
    ObstaclePublisher(const std::string &service, const std::string &instance, const std::string &topic);

    ~ObstaclePublisher();

    void PublishObstacle(const Obstacles &obs);
};

#endif // __OBS_PUBLISHER_H__
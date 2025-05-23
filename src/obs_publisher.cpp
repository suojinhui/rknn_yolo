#include "iostream"
#include "obs_publisher.h"
#include "logger.h"

ObstaclePublisher::ObstaclePublisher(const std::string &service, const std::string &instance, const std::string &topic)
    : topic_info_(TopicInfo(service, instance, topic)),
    publisher_({topic_info_.service_, topic_info_.instance_, topic_info_.topic_}),
    topic_(service + '/' + instance + '/' + topic),
    index_(0)
{
}

ObstaclePublisher::~ObstaclePublisher()
{
}

void ObstaclePublisher::PublishObstacle(const Obstacles &obs)
{

    publisher_.loan()
        .and_then([&](auto &sample){
                    sample->timestamp = obs.timestamp;
                    sample->obs_num = obs.obs_num;
                    std::memcpy(sample->obstacles, obs.obstacles, sizeof(Obstacles));
                    sample.publish();
                    SPDLOG_INFO("topic {} publish timestamp {} success", topic_, obs.timestamp);
                }
            )
        .or_else([&](auto &error)
                 { std::cerr << topic_ + ": Unable to loan sample, error: " << error << std::endl; });
    return;
}
#ifndef __TOPIC_INFO_H__
#define __TOPIC_INFO_H__

#include "iceoryx_posh/runtime/posh_runtime.hpp"
#include "iox/optional.hpp"

/**
 * @class TopicInfo
 * @brief Manages topic metadata and subscriber options for image transport systems.
 *
 * This class stores information about a specific topic, including service,
 * instance, and topic names, which are used to define the scope of communication
 * in a publisher/subscriber pattern. Additionally, it configures subscriber options
 * that affect how data is received.
 */
class TopicInfo
{
public:
    /**
     * @brief Constructs a TopicInfo object with specified service, instance, and topic names.
     *
     * Copies the provided strings into internal character arrays, ensuring
     * that the strings fit within predefined buffer sizes. Also initializes
     * subscriber options with default values.
     *
     * @param service The name of the service.
     * @param instance The instance of the service.
     * @param topic The specific topic name.
     */
    TopicInfo(const std::string &service, const std::string &instance, const std::string &topic)
    {
        strncpy(const_cast<char *>(service_), service.c_str(), sizeof(service_));
        strncpy(const_cast<char *>(instance_), instance.c_str(), sizeof(instance_));
        strncpy(const_cast<char *>(topic_), topic.c_str(), sizeof(topic_));

        // Default subscriber options
        subscriber_options_.queueCapacity = 1U;
        subscriber_options_.historyRequest = 0U;
        subscriber_options_.requiresPublisherHistorySupport = false;
    };

    ~TopicInfo(){};

    char service_[100] = {0};
    char instance_[100] = {0};
    char topic_[100] = {0};
    iox::popo::SubscriberOptions subscriber_options_;
};

#endif // __TOPIC_INFO_H__
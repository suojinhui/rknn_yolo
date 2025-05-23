#ifndef __LOGGER_H__
#define __LOGGER_H__


#include "map"
#include "thread"
#include "mutex"
#include "condition_variable"

#include "spdlog/spdlog.h"

const std::map<std::string, spdlog::level::level_enum> log_level_map = {
        {"trace",    spdlog::level::trace},
        {"debug",    spdlog::level::debug},
        {"info",     spdlog::level::info},
        {"warn",     spdlog::level::warn},
        {"err",      spdlog::level::err},
        {"critical", spdlog::level::critical}};

class Logger {
private:
    std::string log_dir_;
    std::string log_level_;
    int log_retention_days_;

    bool stop_;
    std::mutex stop_mutex_;
    std::condition_variable stop_condition_;

    std::thread clean_thread_;

    Logger(const std::string &log_level, const std::string &log_dir, int log_retention_days);

public:
    static Logger &
    GetInstance(const std::string &log_level = "info", const std::string &log_dir = "./log", int log_retention_days_ = 3);

    ~Logger();

    void CleanLogs();

    void StartCleanThread();

    void Stop();

    Logger(const Logger &) = delete;

    Logger &operator=(const Logger &) = delete;
};

#endif // __LOGGER_H__

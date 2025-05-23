#include "iostream"
#include "filesystem"
#include "map"

#include "spdlog/spdlog.h"
#include "spdlog/sinks/hourly_file_sink.h"

#include "logger.h"
#include "utils.h"

Logger::~Logger() {

}

Logger::Logger(const std::string &log_level,
               const std::string &log_dir,
               int log_retention_days)
        : log_level_(log_level),
          log_dir_(log_dir),
          stop_(false),
          log_retention_days_(log_retention_days) {
    if (!log_level.empty() && log_level_map.find(log_level) != log_level_map.end()) {
        log_level_ = log_level;
    }

    if (!log_dir.empty() && std::filesystem::is_directory(log_dir)) {
        log_dir_ = log_dir;
    }
    if (!std::filesystem::exists(log_dir_)) {
        std::filesystem::create_directories(log_dir_);
    }

    std::string log_file = log_dir_ + "/log.log";
    auto hourly_sink = std::make_shared<spdlog::sinks::hourly_file_sink_mt>(log_file.c_str(), 0, 0);
    auto logger = std::make_shared<spdlog::logger>("hourly_logger", hourly_sink);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e][%^%l%$]%v");

    auto level_it = log_level_map.find(log_level_);
    logger->set_level(level_it->second);
    logger->flush_on(level_it->second);
    spdlog::set_default_logger(logger);

    StartCleanThread();

    SPDLOG_INFO("SPDLOG: Hourly logger Initiated");
}

Logger &Logger::GetInstance(const std::string &log_level, const std::string &log_dir, int expired_days) {
    static Logger logger(log_level, log_dir, expired_days);
    return logger;
}

void Logger::CleanLogs() {
    SPDLOG_INFO("Clean logs");
    const int retention_seconds = log_retention_days_ * 24 * 60 * 60;
    for (const auto &entry: std::filesystem::directory_iterator(log_dir_)) {
        std::cout << entry << std::endl;
        if (!std::filesystem::is_regular_file(entry))
            continue;

        auto file_path = entry.path();
        if (IsFileExpired(file_path, retention_seconds)) {
            SPDLOG_INFO("Removing {}", file_path.string());
            std::cout << "removing " +  file_path.string() << std::endl;
            std::filesystem::remove(file_path);
            continue;
        }

        if (file_path.extension() == ".log" && (stop_ || IsFileExpired(file_path, 5))) {
            CompressFile(file_path);
        }
    }
    return;
}

void Logger::StartCleanThread() {
    clean_thread_ = std::thread([this]() {
        while (true) {
            auto now = std::chrono::system_clock::now();
            auto next_hour = std::chrono::time_point_cast<std::chrono::hours>(now) + std::chrono::hours(1);
            auto target_time = next_hour + std::chrono::seconds(10);
            {
                std::unique_lock<std::mutex> lock(stop_mutex_);
                if (stop_condition_.wait_until(lock, target_time) != std::cv_status::timeout) {
                    if (stop_ == true) {
                        break;
                    }
                    continue;
                }
            }
            CleanLogs();
        }
    });
    return;
}

void Logger::Stop() {
    {
        std::unique_lock<std::mutex> lock(stop_mutex_);
        stop_ = true;
        stop_condition_.notify_one();
    }

    if (clean_thread_.joinable()) {
        clean_thread_.join();
    }

    CleanLogs();
    spdlog::shutdown();
    return;
}

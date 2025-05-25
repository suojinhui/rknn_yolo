#ifndef __LOGGER_H__
#define __LOGGER_H__


#include "map"
#include "thread"
#include "mutex"
#include "condition_variable"

#include "spdlog/spdlog.h"

/**
 * @brief Log level mapping between string representations and spdlog levels
 * 
 * Provides conversion from human-readable log level strings ("debug", "info", etc.)
 * to spdlog's internal level enumeration. Used for configuration validation.
 */
const std::map<std::string, spdlog::level::level_enum> log_level_map = {
        {"trace",    spdlog::level::trace},
        {"debug",    spdlog::level::debug},
        {"info",     spdlog::level::info},
        {"warn",     spdlog::level::warn},
        {"err",      spdlog::level::err},
        {"critical", spdlog::level::critical}};

/**
 * @class Logger
 * @brief Thread-safe logging system with automated log rotation and cleanup
 */
class Logger {
private:
    std::string log_dir_; // Storage directory for log files
    std::string log_level_; // Current logging severity level (e.g., "info", "debug")
    int log_retention_days_; // Maximum age (in days) to keep log files before deletion

    bool stop_; // Flag to signal background thread termination
    std::mutex stop_mutex_; // Mutex for synchronizing access to stop_ flag
    std::condition_variable stop_condition_; // Condition variable for timed cleanup operations

    std::thread clean_thread_; // Background thread handle for log maintenance tasks

    /**
     * @brief Private constructor to enforce singleton pattern
     * @param log_level Initial log severity level
     * @param log_dir Directory path for log storage
     * @param log_retention_days Days to retain log files
     * 
     * @throw std::invalid_argument if invalid log_level provided
     * @throw std::filesystem::filesystem_error on directory creation failure
     */
    Logger(const std::string &log_level, const std::string &log_dir, int log_retention_days);

public:

    /**
     * @brief Get the singleton instance with configurable parameters
     * @param log_level Log severity level (default: "info")
     * @param log_dir Log storage directory (default: "./log")
     * @param expired_days Log retention period in days (default: 3)
     * @return Reference to the singleton Logger instance
     * 
     * @note Subsequent calls to GetInstance will ignore parameters
     *       after the first initialization.
     */
    static Logger &
    GetInstance(const std::string &log_level = "info", const std::string &log_dir = "./log", int log_retention_days_ = 3);

    ~Logger();

    /**
     * @brief Perform immediate log cleanup cycle
     * 
     * Scans log directory and:
     * 1. Deletes files older than retention period
     * 2. Compresses eligible log files
     * 
     * @warning May block on filesystem operations
     */
    void CleanLogs();

    /**
     * @brief Launch background maintenance thread
     * 
     * Starts a daemon thread that automatically:
     * - Performs hourly cleanup at HH:00:10
     * - Handles log rotation events
     * - Manages file compression
     */
    void StartCleanThread();

    /**
     * @brief Gracefully shutdown logging system
     * @note Should be called before program termination
     */
    void Stop();

    Logger(const Logger &) = delete; // Deleted copy constructor
    Logger &operator=(const Logger &) = delete; // Deleted assignment operator
};

#endif // __LOGGER_H__

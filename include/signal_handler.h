#ifndef __SIGNAL_HANDLER_H__
#define __SIGNAL_HANDLER_H__

#include "mutex"
#include "condition_variable"

/**
 * @class SignalHandler
 * @brief Thread-safe signal handling system for graceful shutdown
 * 
 * Implements singleton pattern to capture termination signals (SIGINT/SIGTERM)
 * and coordinate multi-threaded application shutdown. Provides blocking wait
 * primitive for main thread synchronization.
 */
class SignalHandler
{
private:
    bool is_stopped_ = false; // Atomic stop flag indicating signal reception
    std::mutex is_stopped_mutex_; // Mutex guarding access to is_stopped_ flag
    std::condition_variable is_stopped_condition_; // Condition variable for signal notification

    SignalHandler(); // Private constructor enforcing singleton pattern

    /**
     * @brief Internal signal handling callback
     * @param signum Captured signal number (unused)
     * 
     * @warning This handler is async-signal-safe and only modifies atomic flags
     */
    static void Stop();

public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the SignalHandler instance
     */
    static SignalHandler &GetInstance();

    /**
     * @brief Register signal handlers for SIGINT and SIGTERM
     * 
     * Sets up POSIX signal actions to trigger graceful shutdown.
     * Should be called once during application initialization.
     */
    static void SetSignalAction();

    /**
     * @brief Signal handling callback for termination signals
     * 
     * @param signum Received signal number (SIGINT/SIGTERM/SIGHUP)
     */
    static void SignalHandle(int signum);

    /**
     * @brief Blocking wait for termination signal
     * 
     * Suspends calling thread until:
     * 1. SIGINT (Ctrl+C) received
     * 2. SIGTERM received
     * 3. Programmatic shutdown initiated
     */
    void WaitForStopSignal();

    ~SignalHandler(); // Destructor releases system signal handlers

    // Prohibit copying
    SignalHandler(const SignalHandler &) = delete;
    SignalHandler &operator=(const SignalHandler &) = delete;
};

#endif // __SIGNAL_HANDLER_H__
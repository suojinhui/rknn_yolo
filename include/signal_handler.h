#ifndef INCLUDE_SIGNAL_HANDLER_H
#define INCLUDE_SIGNAL_HANDLER_H

#include "mutex"
#include "condition_variable"

class SignalHandler
{
private:
    bool is_stopped_ = false;
    std::mutex is_stopped_mutex_;
    std::condition_variable is_stopped_condition_;

    SignalHandler();

    static void Stop();

public:
    static SignalHandler &GetInstance();

    static void SetSignalAction();

    static void SignalHandle(int signum);

    void WaitForStopSignal();

    ~SignalHandler();

    SignalHandler(const SignalHandler &) = delete;

    SignalHandler &operator=(const SignalHandler &) = delete;
};

#endif // INCLUDE_SIGNAL_HANDLER_H
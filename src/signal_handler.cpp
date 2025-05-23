#include "signal_handler.h"

#include "iostream"
#include "cstring"
#include "signal.h"


SignalHandler::SignalHandler() {}

SignalHandler::~SignalHandler() {}

SignalHandler &SignalHandler::GetInstance() {
    static SignalHandler signal_handler;
    return signal_handler;
}

void SignalHandler::SetSignalAction() {
    struct sigaction sig_action;
    memset(&sig_action, 0, sizeof(sig_action));
    sig_action.sa_handler = SignalHandle;
    sigemptyset(&sig_action.sa_mask);
    sig_action.sa_flags = 0;
    sigaction(SIGINT, &sig_action, NULL);
    sigaction(SIGTERM, &sig_action, NULL);
    sigaction(SIGHUP, &sig_action, NULL);
    return;
}

void SignalHandler::SignalHandle(int signum) {
    switch (signum) {
        case SIGINT:
            std::cout << "QUIT: SIGINT RECEIVED" << std::endl;
            Stop();
            break;
        case SIGTERM:
            std::cout << "QUIT: SIGTERM RECEIVED" << std::endl;
            Stop();
            break;
        case SIGHUP:
            std::cout << "QUIT: SIGHUP RECEIVED" << std::endl;
            Stop();
            break;
        default:
            break;
    }
    return;
}

void SignalHandler::Stop() {
    SignalHandler &signal_handler = GetInstance();
    std::unique_lock<std::mutex> lock(signal_handler.is_stopped_mutex_);
    signal_handler.is_stopped_ = true;
    signal_handler.is_stopped_condition_.notify_one();
    return;
}

void SignalHandler::WaitForStopSignal() {
    std::unique_lock<std::mutex> lock(is_stopped_mutex_);
    is_stopped_condition_.wait(lock, [this] { return is_stopped_; });
    return;
}

#ifndef __CAPTURER_H__
#define __CAPTURER_H__

// Standard and system headers
#include <iostream>
#include <thread>
#include <linux/videodev2.h>
#include <condition_variable>
#include <linux/v4l2-common.h>
#include <linux/v4l2-controls.h>

// OpenCV and image processing headers
#include <opencv2/opencv.hpp>

// Rockchip RGA acceleration headers
#include "im2d.h"
#include "RgaUtils.h"
#include "RgaApi.h"

/**
 * @def TEGRA_CAMERA_CID_BASE
 * @brief Base value for Tegra-specific camera control IDs
 */
#define TEGRA_CAMERA_CID_BASE (V4L2_CTRL_CLASS_CAMERA | 0x2000)

/**
 * @def TEGRA_CAMERA_CID_VI_PREFERRED_STRIDE
 * @brief Control ID for setting preferred memory stride in Tegra cameras
 */
#define TEGRA_CAMERA_CID_VI_PREFERRED_STRIDE (TEGRA_CAMERA_CID_BASE + 110)

/**
 * @def V4L2_BUFFERS_NUM
 * @brief Number of video buffers to allocate for V4L2 streaming
 */
#define V4L2_BUFFERS_NUM 4

/**
 * @def V4L2_PLANES_NUM
 * @brief Number of planes per video buffer (for multi-planar formats)
 */
#define V4L2_PLANES_NUM 1

namespace capturer {

/**
 * @brief Get current timestamp in seconds with millisecond precision
 * @return Double precision timestamp value
 */
double GetCurrentTimestamp();

/**
 * @brief Get current time as formatted string
 * @return String with format "YYYY-MM-DD HH:MM:SS"
 */
std::string GetCurrentTimeString();

/**
 * @brief Multi-planar buffer structure for V4L2 memory mapping
 */
typedef struct {
    void *start[V4L2_PLANES_NUM];  ///< Array of pointers to plane buffers
    size_t length[V4L2_PLANES_NUM]; ///< Array of plane buffer lengths
} MPlaneBuffer;

/**
 * @class Capturer
 * @brief V4L2-based camera capture class with image processing capabilities
 * 
 * Provides hardware-accelerated video capture using V4L2 API and Rockchip RGA
 * for image scaling and transformation. Supports multi-threaded operation
 * with frame rate control and image synchronization.
 */
class Capturer
{
private:
    // Device configuration
    int id_;                 ///< Camera device ID (e.g. /dev/videoX)
    std::string device_;     ///< Device path string
    cv::Size camera_size_;   ///< Native camera resolution
    cv::Size output_size_;   ///< Output image resolution after processing
    int camera_width_;       ///< Native camera width in pixels
    int camera_height_;      ///< Native camera height in pixels
    int output_width_;       ///< Output image width in pixels
    int output_height_;      ///< Output image height in pixels

    // V4L2 device handles
    int fd_;                ///< File descriptor for V4L2 device
    unsigned int pixfmt_;    ///< Current pixel format (V4L2_PIX_FMT_*)
    MPlaneBuffer* buffer_cam_ = nullptr; ///< Mapped buffer array

    // Capture state
    int frame_count_;       ///< Frame counter for FPS calculation
    bool stop_;             ///< Flag to stop capture thread
    std::thread capture_thread_; ///< Capture thread handle
    std::mutex mutex_;      ///< General-purpose mutex for data protection

    // Performance tracking
    int fps_;               ///< Target frames per second
    int actual_fps_;        ///< Measured actual frames per second
    int flip_method_;       ///< Image flip method (0-3, see V4L2 docs)
    std::mutex fps_mutex_;  ///< Mutex for FPS data synchronization

    // Image data
    cv::Mat image_;         ///< Current captured image (OpenCV matrix)
    std::string label_;     ///< Device label for identification
    
public:
    // Image synchronization
    std::mutex image_ready_mutex;       ///< Mutex for image access
    bool data_ready = false;            ///< Flag indicating new frame availability
    std::condition_variable image_ready_condition_; ///< Frame ready notification

    // Initialization status
    bool initiated_ = false; ///< Flag indicating successful initialization

public:
    /**
     * @brief Destructor - Releases all allocated resources
     */
    ~Capturer();

    /**
     * @brief Constructor - Initializes capture parameters
     * @param id Camera device ID (0 for /dev/video0)
     * @param camera_width Native camera sensor width
     * @param camera_height Native camera sensor height
     * @param output_width Output image width after processing
     * @param output_height Output image height after processing
     * @param fps Target frame rate (frames per second)
     * @param flip_method Image flip method (0=none, 1=horizontal, etc.)
     * @param label Device identifier string
     */
    Capturer(int id, int camera_width, int camera_height, int output_width, int output_height, 
                int fps, int flip_method, const std::string &label);

    /**
     * @brief Initializes V4L2 device parameters
     * 
     * Sets up pixel format, resolution, and streaming parameters
     * using V4L2 ioctl calls. Must be called before capture.
     */
    void InitV4L2();

    /**
     * @brief Requests video buffers from V4L2 device
     * 
     * Negotiates buffer allocation with the V4L2 driver using
     * VIDIOC_REQBUFS command. Prepares for memory mapping.
     */
    void RequestCameraBuffer();

    /**
     * @brief Maps V4L2 buffers to user space memory
     * 
     * Creates memory mappings for all requested buffers using
     * VIDIOC_QUERYBUF and mmap. Allocates MPlaneBuffer structures.
     */
    void PrepareBuffers();

    /**
     * @brief Starts video streaming from the device
     * 
     * Enqueues all buffers and starts streaming using VIDIOC_STREAMON.
     * After this call, frames will be captured into buffers.
     */
    void StartStream();

    /**
     * @brief Main capture loop (blocking)
     * 
     * Continuously dequeues buffers, processes images (scaling/flipping),
     * and updates the current image. Uses RGA hardware acceleration.
     */
    void StartCaptureLoop();

    /**
     * @brief Starts capture thread
     * 
     * Launches a new thread running StartCaptureLoop()
     */
    void StartCaptureThread();

    /**
     * @brief Stops video streaming
     * 
     * Issues VIDIOC_STREAMOFF command and cancels buffer queue
     */
    void StopStream();

    /**
     * @brief Stops capture thread and cleans up resources
     */
    void StopCapture();

    /**
     * @brief Calculates actual frame rate
     * 
     * Measures frames captured over time to update actual_fps_
     */
    void SpyOnFps();

    /**
     * @brief Full resource cleanup
     * 
     * Releases all buffers, closes device handle, and resets state.
     * Called automatically by destructor.
     */
    void Destroy();

    /**
     * @brief Retrieves current captured image
     * @return cv::Mat containing latest frame
     * @note Blocks until image_ready_condition_ is notified
     */
    cv::Mat GetImage();
};

} // namespace capturer

#endif // __CAPTURER_H__
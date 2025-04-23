#include "iostream"
#include "thread"
#include "mutex"
#include "condition_variable"
#include "filesystem"

#include "csignal"
#include "fcntl.h"
#include "sys/ioctl.h"
#include "sys/poll.h"
#include "sys/mman.h"
#include "linux/v4l2-common.h"
#include "linux/v4l2-controls.h"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "RgaApi.h"

#include "capturer.h"

namespace capturer {

double GetCurrentTimestamp()
{
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        return 0;
    }
    return ((double)time.tv_sec + (double)time.tv_usec * .000001) * 1000;
}

std::string GetCurrentTimeString()
{
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::tm timeInfo;
    localtime_r(&time, &timeInfo);

    char time_string[26];
    std::strftime(time_string, sizeof(time_string), "%Y_%m_%d_%H_%M_%S", &timeInfo);
    return time_string;
}


Capturer::Capturer(int id, int camera_width, int camera_height, int output_width, int output_height, int fps, 
                 int flip_method, const std::string &label):
                 id_(id), 
                 camera_width_(camera_width), 
                 camera_height_(camera_height), 
                 output_width_(output_width), 
                 output_height_(output_height), 
                 fps_(fps), 
                 actual_fps_(0), 
                 flip_method_(flip_method), 
                 device_("/dev/video" + std::to_string(id_)), 
                 pixfmt_(V4L2_PIX_FMT_NV12), 
                 stop_(false), 
                 frame_count_(0), 
                 label_(label), 
                 image_(output_height_, output_width_, CV_8UC3)
                 {}

Capturer::~Capturer() {}

void Capturer::InitV4L2(){
    // Open device
    fd_ = open(device_.c_str(), O_RDWR);
    if (-1 == fd_)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to open this device.", id_);
        return;
    }

    // init RGA
    if (c_RkRgaInit() < 0) {
        SPDLOG_ERROR("Capturer_{}: RGA init failed", id_);
        return;
    }

    // Set Camera output format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = camera_width_;
    fmt.fmt.pix_mp.height = camera_height_;
    fmt.fmt.pix_mp.pixelformat = pixfmt_;
    fmt.fmt.pix_mp.num_planes  = V4L2_PLANES_NUM;
    fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;
    
    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to set output format, camera may not have been released", id_);
        return;
    }

    // Get the real format in case the desired is not supported
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(fd_, VIDIOC_G_FMT, &fmt) < 0)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to get output format.", id_);
        return;
    }

    if (fmt.fmt.pix.width != camera_width_ || fmt.fmt.pix.height != camera_height_ || fmt.fmt.pix.pixelformat != pixfmt_)
    {
        camera_width_ = fmt.fmt.pix.width;
        camera_height_ = fmt.fmt.pix.height;
        pixfmt_ = fmt.fmt.pix.pixelformat;
        SPDLOG_WARN("Capturer_{}: Camera may not support desired output format, use {}x{} {} instead.", 
                            id_, camera_width_, camera_height_, pixfmt_);
    }

    SPDLOG_INFO("Capturer_{}: Succeed in initializing v4l2 components.", id_);
    initiated_ = true;
    return;
}

void Capturer::RequestCameraBuffer(){

    struct v4l2_requestbuffers request_buffers;
    memset(&request_buffers, 0, sizeof(request_buffers));
    request_buffers.count = V4L2_BUFFERS_NUM;
    request_buffers.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    request_buffers.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &request_buffers) < 0)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to request v4l2 buffers.", id_);
        Destroy();
        exit(-1);
    }
    if (V4L2_BUFFERS_NUM != request_buffers.count)
    {
        SPDLOG_ERROR("Capturer_{}: V4l2 buffers number is not as desired.", id_);
        Destroy();
        exit(-1);
    }

    for (size_t index = 0; index < V4L2_BUFFERS_NUM; index++)
    {
        struct v4l2_buffer buffer;
        struct v4l2_plane planes[V4L2_PLANES_NUM];
        memset(&buffer, 0, sizeof(buffer));
        memset(planes, 0, sizeof(planes));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.m.planes = planes;
        buffer.length = V4L2_PLANES_NUM;
        buffer.index = index;
        
        // Query v4l2 buffer
        if (ioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0)
        {
            SPDLOG_ERROR("Capturer_{}: Failed to query v4l2 buffer.", id_);
            Destroy();
            exit(-1);
        }

        for (unsigned int i = 0; i < buffer.length; i++) {
            buffer_cam_[index].length[i] = planes[i].length;
            buffer_cam_[index].start[i] = mmap(
                NULL,
                planes[i].length,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd_,
                planes[i].m.mem_offset
            );

            if (buffer_cam_[index].start[i] == MAP_FAILED) {
                SPDLOG_ERROR("Capturer_{}: Mmap failed for v4l2 buffer plane {}", i);
                exit(EXIT_FAILURE);
            }
        }

        if (ioctl(fd_, VIDIOC_QBUF, &buffer) < 0)
        {
            SPDLOG_ERROR("Capturer_{}: Failed to queue v4l2 buffer.", id_);
            Destroy();
            exit(-1);
        }
    }

    SPDLOG_INFO("Capturer_{}: Succeed in requesting v4l2 buffers.", id_);
    return;
}

void Capturer::PrepareBuffers()
{
    // Allocate buffer context
    buffer_cam_ = (MPlaneBuffer*) malloc(V4L2_BUFFERS_NUM * sizeof(MPlaneBuffer));
    memset(buffer_cam_, 0, V4L2_BUFFERS_NUM * sizeof(MPlaneBuffer));

    if (NULL == buffer_cam_)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to allocate global nv_buffer context.", id_);
        Destroy();
        exit(-1);
    }

    RequestCameraBuffer();

    SPDLOG_INFO("Capturer_{}: Succeed in preparing stream buffers.", id_);
    return;
}

void Capturer::StartStream()
{   
    // Start v4l2 streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0)
    {
        SPDLOG_ERROR("Capturer_{}: Failed to start streaming.", id_);
        Destroy();
        exit(-1);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    SPDLOG_INFO("Capturer_{}: Succeed in starting streaming.", id_);
    return;
}

void Capturer::StartCaptureLoop()
{
    SPDLOG_INFO("Capturer_{}: Start capture loop.", id_);
    struct pollfd poll_fds[1];
    memset(&poll_fds[0], 0, sizeof(pollfd));
    poll_fds[0].fd = fd_;
    poll_fds[0].events = POLLIN;

    // TODO：rga buffer and image ptr should be init in a other function
    struct v4l2_buffer buffer;
    struct v4l2_plane planes[V4L2_PLANES_NUM];
    memset(&buffer, 0, sizeof(buffer));
    memset(planes, 0, sizeof(planes));
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.m.planes = planes;
    buffer.length = V4L2_PLANES_NUM;

    uint8_t *nv12_buffer = new uint8_t[camera_width_ * camera_height_ * 3 / 2];
    uint8_t *scaled_nv12_buffer = new uint8_t[output_width_ * output_height_ * 3 / 2];

    uint8_t* image_buffer;

    int src_format = RK_FORMAT_YCrCb_420_SP; 
    int dst_format = RK_FORMAT_RGB_888;      

    size_t nv12_size = output_width_ * output_height_ * get_bpp_from_format(src_format); 
    size_t bgr_size = output_width_ * output_height_ * get_bpp_from_format(dst_format);         

    uint8_t* bgr_buf = (uint8_t*)malloc(bgr_size);
    image_.data = bgr_buf;

    rga_buffer_t g_src_img;
    rga_buffer_t g_dst_img;
    memset(&g_src_img, 0, sizeof(g_src_img));
    memset(&g_dst_img, 0, sizeof(g_dst_img));

    rga_buffer_handle_t g_src_handle;
    rga_buffer_handle_t g_dst_handle;
    g_src_handle = 0;
    g_dst_handle = 0;
    
    g_src_handle = importbuffer_virtualaddr(scaled_nv12_buffer,  nv12_size);
    g_dst_handle = importbuffer_virtualaddr(bgr_buf,  bgr_size);
    if (g_src_handle == 0 || g_dst_handle == 0)
    {
        SPDLOG_ERROR("Capturer_{}: RGA importbuffer failed!", id_);
        Destroy();
        exit(-1);
        return ;
    }

    g_src_img = wrapbuffer_handle(g_src_handle, output_width_, output_height_, src_format);
    g_dst_img = wrapbuffer_handle(g_dst_handle, output_width_, output_height_, dst_format);

    IM_STATUS checkRet = imcheck(g_src_img, g_dst_img, {}, {});
    if (IM_STATUS_NOERROR != checkRet)
    {
        SPDLOG_ERROR("Capturer_{}: RGA check buffer error!", id_);
        Destroy();
        exit(-1);
        return;
    }
    
    double start_time = 0.;
    double stop_time = 0.;
    
    std::thread spy_on_fps_thread = std::thread(&Capturer::SpyOnFps, this);
    
    while (poll(poll_fds, 1, 5000) > 0 && !stop_)
    {   
        if (poll_fds[0].revents & POLLIN)
        {
            start_time = GetCurrentTimestamp();
           
            // Dequeue camera buffer
            if (ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0)
            {
                SPDLOG_ERROR("Capturer_{}: Failed to dequeue camera buffer.", id_);
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(fps_mutex_);
                ++actual_fps_;
            }
            
            image_buffer = (uint8_t*)buffer_cam_[buffer.index].start[0];

            memcpy(scaled_nv12_buffer, image_buffer, nv12_size);

            // NV12 to BGR (RGA)
            int ret = imcvtcolor(g_src_img, 
                g_dst_img, 
                src_format, 
                dst_format);


            if (ret != IM_STATUS_SUCCESS) {
                SPDLOG_ERROR("Capturer_{}: Convert color failed.", id_);
                Destroy();
                exit(-1);
            }
            
            {
                std::unique_lock<std::mutex> lock(image_ready_mutex);
                data_ready = true;
                image_ready_condition_.notify_all();
            }
            
            // Enqueue camera buff
            if (ioctl(fd_, VIDIOC_QBUF, &buffer))
            {
                SPDLOG_ERROR("Capturer_{}: Failed to queue camera buffer.", id_);
                Destroy();
                exit(-1);
            }

            stop_time = GetCurrentTimestamp();
            // SPDLOG_INFO("Capturer_{}: frame {} time = {} ms", id_, frame_count_, stop_time - start_time);
            ++frame_count_;
        }
    }

    if (spy_on_fps_thread.joinable())
    {
        spy_on_fps_thread.join();
    }
    image_buffer = nullptr;

    uint8_t *release_pointers[] = {scaled_nv12_buffer, bgr_buf};

    for (uint8_t *&ptr : release_pointers)
    {
        if (ptr)
        {
            delete[] ptr;
            ptr = nullptr;
        }
    }
    if (g_src_handle) releasebuffer_handle(g_src_handle);
    if (g_dst_handle) releasebuffer_handle(g_dst_handle);
    return;
}

void Capturer::StartCaptureThread()
{
    InitV4L2();

    if (!initiated_)
    {
        raise(SIGINT);
        return;
    }

    PrepareBuffers();

    StartStream();

    capture_thread_ = std::thread(&Capturer::StartCaptureLoop, this);
    return;
}

void Capturer::StopStream()
{
    // Stop v4l2 streaming
    if (!initiated_)
        return;

    SPDLOG_INFO("Capturer_{}: Stop capture loop.", id_);
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type))
    {
        SPDLOG_ERROR("Capturer_{}: Failed to stop streaming", id_);
        return;
    }

    SPDLOG_INFO("Capturer_{}: Succeed in stopping streaming", id_);
    return;
}

void Capturer::StopCapture()
{
    StopStream();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }

    actual_fps_ = -1;

    if (capture_thread_.joinable())
    {
        capture_thread_.join();
    }

    Destroy();

    SPDLOG_INFO("Capturer_{}: Stopped", id_);

    return;
}

void Capturer::SpyOnFps()
{
    int id;
    while (!stop_)
    {
        id = id_;
        {
            std::lock_guard<std::mutex> lock(fps_mutex_);
            actual_fps_ = 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        {
            std::lock_guard<std::mutex> lock(fps_mutex_);
            if (actual_fps_ != -1 && abs(actual_fps_ - fps_) > 1)
            {
                if (stop_)
                {
                    actual_fps_ = -1;
                    return;
                }
                std::stringstream ss;
                ss << "Capturer_" << id << " fps = " << actual_fps_;
                std::cout << ss.str() << std::endl;
                SPDLOG_WARN("Capturer_{}: WRANING! FPS has decreased, FPS = {}", id, actual_fps_);
            }
            actual_fps_ = 0;
        }
    }

    return;
}

void Capturer::Destroy()
{
    actual_fps_ = -1;

    if (fd_ > 0)
    {
        close(fd_);
    }

    if (buffer_cam_ != nullptr)
    {
        for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++)
        {
            auto &buffer = buffer_cam_[i];
            for (int j = 0; j < V4L2_PLANES_NUM; j++) {
                munmap(buffer.start[j], buffer.length[j]);
            }
        }
        free(buffer_cam_);
        buffer_cam_ = nullptr;   
    }
    SPDLOG_INFO("Capturer_{}: Succeed in destroying v4l2 components.", id_);
    return;
}

cv::Mat Capturer::GetImage()
{
    {
        std::unique_lock<std::mutex> lock(image_ready_mutex);
        image_ready_condition_.wait(lock, [this]()
                                    { return data_ready; });
        data_ready = false;
    }
    return image_;
}

} // namespace capturer
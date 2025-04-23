#include "model.h"
#include "spdlog/spdlog.h"
#include "utils.h"

YOLO_MODEL::YOLO_MODEL(const char* model_name,  int num_classes, float nms_threshold, float box_conf_threshold_, 
    int input_width, int input_height, Datasets dataset_type):
    model_name_(model_name),
    num_classes_(num_classes),
    nms_threshold_(nms_threshold),
    image_info_input_(image_info(input_height, input_width, 3)),
    box_conf_threshold_(box_conf_threshold_) 
    {
        if(dataset_type == Datasets::COCO)
        {
            labels_map_ = shared_ptr<IBaseLabels>(new CocoLabels());
        } else if (dataset_type == Datasets::Bdd100k)
        {
            labels_map_ = shared_ptr<IBaseLabels>(new Bdd100kLabels());
        } else {
            SPDLOG_ERROR("Unsupported dataset type!");
            std::abort();
        }
    }

YOLO_MODEL::~YOLO_MODEL() {}

bool YOLO_MODEL::load_model() {
    int ret;
    int model_size = 0;
    FILE *fp;
    unsigned char *model_data;

    SPDLOG_INFO("Loading model...");

    // read model from .rknn file
    fp = fopen(model_name_, "rb");
    if (NULL == fp)
    {
        SPDLOG_ERROR("Open file {} failed.", model_name_);
        return false;
    }
    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);

    ret = fseek(fp, 0, SEEK_SET);
    if (ret != 0)
    {
        SPDLOG_ERROR("blob seek failure.");
        return false;
    }
    model_data = (unsigned char *)malloc(model_size);
    if (model_data == NULL)
    {
        SPDLOG_ERROR("buffer malloc failure.");
        return false;
    }
    ret = fread(model_data, 1, model_size, fp);
    fclose(fp);

    if (model_data == NULL)
    {
        return false;
    }

    // rknn init
    ret = rknn_init(&model_rk_context_, model_data, model_size, 0, NULL);
    if (ret < 0)
    {
        SPDLOG_ERROR("rknn_init error ret={}", ret);
        return false;
    }

    // rknn query version, io_nums, io_attrs, fmt, etc
    ret = rknn_query(model_rk_context_, RKNN_QUERY_SDK_VERSION, &version_, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        SPDLOG_ERROR("rknn_query RKNN_QUERY_SDK_VERSION error ret={}", ret);
        return false;
    }
    SPDLOG_INFO("sdk version: {} driver version: {}", version_.api_version, version_.drv_version);

    ret = rknn_query(model_rk_context_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(rknn_input_output_num));
    if (ret < 0)
    {
        SPDLOG_ERROR("rknn_query RKNN_QUERY_IN_OUT_NUM error ret=%d\n", ret);
        return false;
    }
    SPDLOG_INFO("model input num: {}, output num: {}", io_num_.n_input, io_num_.n_output);

    input_attrs_ = (rknn_tensor_attr *)malloc(io_num_.n_input * sizeof(rknn_tensor_attr));
    memset(input_attrs_, 0, io_num_.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num_.n_input; i++)
    {
        input_attrs_[i].index = i;
        ret = rknn_query(model_rk_context_, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
        SPDLOG_ERROR("rknn_query RKNN_QUERY_INPUT_ATTR error ret=%d\n", ret);
        return false;
        }
        // default input type is int8 (normalize and quantize need compute in outside)
        // if set uint8, will fuse normalize and quantize to npu
        input_attrs_[i].type = RKNN_TENSOR_UINT8;
        dump_tensor_attr(&(input_attrs_[i]));
    }

    output_attrs_ = (rknn_tensor_attr *)malloc(io_num_.n_output * sizeof(rknn_tensor_attr));
    memset(output_attrs_, 0, io_num_.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num_.n_output; i++)
    {
        output_attrs_[i].index = i;
        // The use of RKNN_QUERY_CATIVE_OTPUT_ATTR in zero copy API will identify the output format with the most hardware 
        // advantages, which may be NC1HWC2. Its arrangement takes into account memory hardware, and the output needs to be 
        // converted from NC1HWC2 to NHWC to adapt to post-processing. The inference comparison here has not been conducted, 
        // please refer to the official API documentation for details
        ret = rknn_query(model_rk_context_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
        SPDLOG_ERROR("rknn_query RKNN_QUERY_OUTPUT_ATTR error ret={}", ret);
        return false;
        }
        out_scales_.push_back(output_attrs_[i].scale);
        out_zps_.push_back(output_attrs_[i].zp);
        
        dump_tensor_attr(&(output_attrs_[i]));
    }

    input_format_ = input_attrs_[0].fmt;

    if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW)
    {
        SPDLOG_ERROR("npu only support NHWC in zero copy mode");
        return false;
    }
    else
    {
        SPDLOG_INFO("model is NHWC input fmt");
        image_info_.h = input_attrs_[0].dims[1];
        image_info_.w = input_attrs_[0].dims[2];
        image_info_.c = input_attrs_[0].dims[3];
    }
    SPDLOG_INFO("model input height={}, width={}, channel={}", image_info_.h, image_info_.w, image_info_.c);

    // Enable the multi-core model of rk3588's NPU to improve inference speed
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    // rknn_core_mask core_mask;
    // switch (get_core_num())
    // {
    // case 0:
    //     core_mask = RKNN_NPU_CORE_0;
    //     break;
    // case 1:
    //     core_mask = RKNN_NPU_CORE_1;
    //     break;
    // case 2:
    //     core_mask = RKNN_NPU_CORE_2;
    //     break;
    // }
    ret = rknn_set_core_mask(model_rk_context_, core_mask);
    if (ret < 0)
    {
        SPDLOG_ERROR("rknn_ctx set mutil core error, ret={}", ret);
        return false;
    }

    return true;
}

bool YOLO_MODEL::init_rga_buffer() {
    
    src_handle_ = 0;
    dst_handle_ = 0;
    memset(&src_, 0, sizeof(src_));
    memset(&dst_, 0, sizeof(dst_));

    image_  = new uint8_t[image_info_input_.w * image_info_input_.h * 3];
    src_handle_ = importbuffer_virtualaddr((void *)image_,  image_info_input_.w * image_info_input_.h * get_bpp_from_format(RK_FORMAT_RGB_888));
    // the input fd of the rknn model is registered in rga to achieve zero copy of image scaling and model inference
    dst_handle_ = importbuffer_fd(model_inputs_[0]->fd,  image_info_.w * image_info_.h * get_bpp_from_format(RK_FORMAT_RGB_888));

    if (src_handle_ == 0 || dst_handle_ == 0)
    {
        SPDLOG_ERROR("RGA importbuffer failed!");
        return false;
    }

    src_ = wrapbuffer_handle(src_handle_, image_info_input_.w, image_info_input_.h, RK_FORMAT_RGB_888);
    dst_ = wrapbuffer_handle(dst_handle_, image_info_.w, image_info_.h, RK_FORMAT_RGB_888);

    int ret = imcheck(src_, dst_, {}, {});
    if (IM_STATUS_NOERROR != ret)
    {
        SPDLOG_ERROR("rga check error! {}", imStrError((IM_STATUS)ret));
        return false;
    }

    SPDLOG_INFO("rga buffer init success!");
    return true;
}

bool YOLO_MODEL::init_io_buffer() {
    // For the input rknn_mam, its size is determined using input_trtrs_[i].size_with_stride 
    // to accommodate hardware memory alignment requirements
    for (uint32_t i = 0; i < io_num_.n_input; i++) // in fact, only one input
    {
        model_inputs_[i] = rknn_create_mem(model_rk_context_, input_attrs_[i].size_with_stride);
        int ret = rknn_set_io_mem(model_rk_context_, model_inputs_[i], &input_attrs_[i]);
        if (ret < 0) {
            SPDLOG_ERROR("rknn_set_input_mem fail! ret={}", ret);
            return false;
        }
    }
    // For output, rknn_create_mem() allocates memory using the number of bytes of the data type 
    // filled by the user * n_elems. Here, the output is int8, which can be taken as input_attrs_[i].size 
    // For the output that requires fp32, i.e., output inverse quantization is performed internally in the 
    // model, which is sizeof(float) * n_elems.
    for (uint32_t i = 0; i < io_num_.n_output; i++) {
        int output_size = output_attrs_[i].size;
        model_outputs_[i] = rknn_create_mem(model_rk_context_, output_size);
        int ret = rknn_set_io_mem(model_rk_context_, model_outputs_[i], &output_attrs_[i]);
        if (ret < 0) {
            SPDLOG_ERROR("rknn_set_output_mem fail! ret={}", ret);
            return false;
        }
    }

    SPDLOG_INFO("init io buffer success!");
    return true;
}


bool YOLO_MODEL::preprocess(cv::Mat& orig_img) {

    // BGR -> RGB
    cv::Mat image;
    cv::cvtColor(orig_img, image, cv::COLOR_BGR2RGB);
    
    // image copy
    memcpy(image_, image.data, image_info_input_.w * image_info_input_.h * 3);

    // resize image
    int ret = imresize(src_, dst_);
    if (IM_STATUS_SUCCESS != ret)
    {
        SPDLOG_ERROR("rga resize error! ret={}", ret);
        return false;
    }
    return true;
}

void YOLO_MODEL::postprocess() {

    // detect results pool
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // stride 8
    int stride0 = 8;
    int grid_h0 = image_info_.h / stride0;
    int grid_w0 = image_info_.w / stride0;
    int validCount0 = 0;
    validCount0 = process((int8_t*) model_outputs_[0]->virt_addr, (int *)anchor0, grid_h0, grid_w0, image_info_.h, image_info_.w, stride0, filterBoxes, objProbs,
                            classId, box_conf_threshold_, out_zps_[0], out_scales_[0], num_classes_);

    // stride 16
    int stride1 = 16;
    int grid_h1 = image_info_.h / stride1;
    int grid_w1 = image_info_.w / stride1;
    int validCount1 = 0;
    validCount1 = process((int8_t*) model_outputs_[1]->virt_addr, (int *)anchor1, grid_h1, grid_w1, image_info_.h, image_info_.w, stride1, filterBoxes, objProbs,
                            classId, box_conf_threshold_, out_zps_[1], out_scales_[1], num_classes_);

    // stride 32
    int stride2 = 32;
    int grid_h2 = image_info_.h / stride2;
    int grid_w2 = image_info_.w / stride2;
    int validCount2 = 0;
    validCount2 = process((int8_t*) model_outputs_[2]->virt_addr, (int *)anchor2, grid_h2, grid_w2, image_info_.h, image_info_.w, stride2, filterBoxes, objProbs,
                            classId, box_conf_threshold_, out_zps_[2], out_scales_[2], num_classes_);

    int validCount = validCount0 + validCount1 + validCount2;

    // no object detect
    if (validCount <= 0)
    {
        return;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i)
    {
        indexArray.push_back(i);
    }
    // sort for nms
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    // class-aware nms
    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold_);
    }

    int last_count = 0;
    detect_result_group_.count = 0;
    float scale_w = (float)image_info_.w / image_info_input_.w;
    float scale_h = (float)image_info_.h / image_info_input_.h;

    // Organize the detection results and input the scaled corresponding model into the image size
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
        continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        detect_result_group_.results[last_count].box.left = (int)(clamp(x1, 0, image_info_.w) / scale_w);
        detect_result_group_.results[last_count].box.top = (int)(clamp(y1, 0, image_info_.h) / scale_h);
        detect_result_group_.results[last_count].box.right = (int)(clamp(x2, 0, image_info_.w) / scale_w);
        detect_result_group_.results[last_count].box.bottom = (int)(clamp(y2, 0, image_info_.h) / scale_h);
        detect_result_group_.results[last_count].prop = obj_conf;
        detect_result_group_.results[last_count].id = id;

        last_count++;
    }
    detect_result_group_.count = last_count;

    return;
}

void YOLO_MODEL::reset_task() {
    memset(&detect_result_group_, 0, sizeof(detect_result_group_t));
}

cv::Mat YOLO_MODEL::inference(cv::Mat& orig_img) {
    reset_task();
    struct timeval start_time, stop_time;
    
    gettimeofday(&start_time, NULL);
    // preprocess
    if(!preprocess(orig_img)){
        SPDLOG_ERROR("preprocess error!");
        destroy();
        std::abort();
    }
    gettimeofday(&stop_time, NULL);
    SPDLOG_DEBUG("preprocess(BGR2RGB+resize): {} ms", (_get_us(stop_time) - _get_us(start_time)) / 1000);

    // inference
    gettimeofday(&start_time, NULL);
    int ret = rknn_run(model_rk_context_, NULL);
    if (ret < 0)
    {
        SPDLOG_ERROR("rknn_run error ret={}", ret);
        destroy();
        std::abort();
    }
    gettimeofday(&stop_time, NULL);
    SPDLOG_DEBUG("inference(npu): {} ms", (_get_us(stop_time) - _get_us(start_time)) / 1000);

    // postprocess
    gettimeofday(&start_time, NULL);
    postprocess();
    gettimeofday(&stop_time, NULL);
    SPDLOG_DEBUG("postprocess(deqnt+nms): {} ms", (_get_us(stop_time) - _get_us(start_time)) / 1000);

    // draw result
    // TODO: Can be encapsulated as a function 
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    float font_scale = 0.001 * std::min(orig_img.cols, orig_img.rows);
    int font_thickness = 1;
    int baseline = 0;
    SPDLOG_DEBUG("------------------num_object:{}----------------------",detect_result_group_.count);
    for (int i = 0; i < detect_result_group_.count; i++)
    {
        detect_result_t* det_result = &(detect_result_group_.results[i]);
        auto name      = labels_map_->get_label(det_result->id);
        auto rec_color = labels_map_->get_color(det_result->id);
        auto txt_color = labels_map_->get_inverse_color(rec_color);

        auto txt = cv::format({"%s: %.2f%%"}, name.c_str(), det_result->prop * 100);
        auto txt_size = cv::getTextSize(txt, font_face, font_scale, font_thickness, &baseline);

        int txt_height = txt_size.height + baseline + 10;
        int txt_width  = txt_size.width + 3;
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;


        cv::Point txt_pos(round(x1), round(y1 - (txt_size.height - baseline + font_thickness)));
        cv::Rect  txt_rec(round(x1 - font_thickness), round(y1 - txt_height), txt_width, txt_height);
        cv::Rect  box_rec(round(x1), round(y1), round(x2 - x1), round(y2 - y1));

        cv::rectangle(orig_img, box_rec, rec_color, 3);
        cv::rectangle(orig_img, txt_rec, rec_color, -1);
        cv::putText(orig_img, txt, txt_pos, font_face, font_scale, txt_color, font_thickness, 16);
        SPDLOG_DEBUG("{} @ (Confidence: {}, Box[ltrb]: {} {} {} {})", name, det_result->prop, det_result->box.left, 
            det_result->box.top, det_result->box.right, det_result->box.bottom);
    }
    return orig_img;

}

void YOLO_MODEL::destroy() {

    // detele rknn io mem
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        if (model_inputs_[i]) {
            rknn_destroy_mem(model_rk_context_, model_inputs_[i]);
        }
    }
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        if (model_outputs_[i]) {
            rknn_destroy_mem(model_rk_context_, model_outputs_[i]);
        }
    }

    // destroy rknn
    rknn_destroy(model_rk_context_);

    if (image_) {
        delete[] image_;
        image_ = nullptr;
    }

    // detele rga buffer
    if (src_handle_)
        releasebuffer_handle(src_handle_);
    if (dst_handle_)
        releasebuffer_handle(dst_handle_);

    SPDLOG_INFO("yolo model destory!");

}

void YOLO_MODEL::init_model() {
    // create model
    if (!load_model()) {
        SPDLOG_ERROR("load model error!");
        destroy();
        std::abort();
    }

    // set io buffer
    if (!init_io_buffer()) {
        SPDLOG_ERROR("init io buffer error!");
        destroy();
        std::abort();
    }

    // init rga, and link input buffer to rga output buffer in order to no copy
    if (!init_rga_buffer()) {
        SPDLOG_ERROR("init rga buffer error!");
        destroy();
        std::abort();
    }
}

std::shared_ptr<YOLO_MODEL> make_detector(const char* model_name, int num_classes, float nms_threshold, 
    float box_conf_threshold, int input_width, int input_height, Datasets dataset_type) {
        return make_shared<YOLO_MODEL>(model_name, num_classes, nms_threshold, box_conf_threshold, input_width, 
            input_height, dataset_type);
    }
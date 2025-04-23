#ifndef __RKNNPOOL_H__
#define __RKNNPOOL_H__

#include "ThreadPool.h"
#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>

#include "label.h"

// rknnModel: YOLO_MODEL, inputType: cv::Mat, outputType: cv::Mat
template <typename rknnModel, typename inputType, typename outputType>
class rknnPool
{
private:
    int threadNum;
    const char* model_name;
    int num_classes; 
    float nms_threshold;
    float box_conf_threshold;
    int input_width;
    int input_height;
    Datasets dataset_type;

    long long id;
    std::mutex idMtx, queueMtx;
    std::unique_ptr<dpool::ThreadPool> pool;
    std::queue<std::future<outputType>> futs;
    std::vector<std::shared_ptr<rknnModel>> models;

protected:
    int getModelId();

public:
    rknnPool(const char* model_name, int num_classes, float nms_threshold, 
        float box_conf_threshold, int input_width, int input_height, Datasets dataset_type, int threadNum);
    int init();

    // Model inference
    int put(inputType inputData);

    // Get the results of your inference
    int get(outputType &outputData);
    ~rknnPool();
};

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::rknnPool(const char* model_name, int num_classes, float nms_threshold, 
    float box_conf_threshold, int input_width, int input_height, Datasets dataset_type, int threadNum)
{
    this->model_name = model_name;
    this->num_classes = num_classes;
    this->nms_threshold = nms_threshold;
    this->box_conf_threshold = box_conf_threshold;
    this->input_width = input_width;
    this->input_height = input_height;
    this->dataset_type = dataset_type;
    this->threadNum = threadNum;
    this->id = 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::init()
{
    try
    {
        this->pool = std::make_unique<dpool::ThreadPool>(this->threadNum);
        for (int i = 0; i < this->threadNum; i++)
            models.push_back(std::make_shared<rknnModel>(this->model_name, this->num_classes, this->nms_threshold, 
                this->box_conf_threshold, this->input_width, this->input_height, this->dataset_type));
    }
    catch (const std::bad_alloc &e)
    {
        std::cout << "Out of memory: " << e.what() << std::endl;
        return -1;
    }
    // Initialize the model
    for (int i = 0, ret = 0; i < threadNum; i++)
    {
        models[i]->init_model(); // TODO: design reuse ctx
        
    }

    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::getModelId()
{
    std::lock_guard<std::mutex> lock(idMtx);
    int modelId = id % threadNum;
    id++;
    return modelId;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
    futs.push(pool->submit(&rknnModel::inference, models[this->getModelId()], inputData));
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::get(outputType &outputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);
    if(futs.empty() == true)
        return 1;
    outputData = futs.front().get();
    futs.pop();
    return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::~rknnPool()
{
    while (!futs.empty())
    {
        outputType temp = futs.front().get();
        futs.pop();
    }
    for (int i = 0; i < threadNum; i++)
    {
        models[i]->destroy();
    }
}

#endif // __RKNNPOOL_H__
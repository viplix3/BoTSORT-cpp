#pragma once

#include <opencv2/core.hpp>

#include "DataType.h"
#include "ReIDParams.h"

class ReIDModel
{
public:
    ReIDModel(const ReIDParams &params, const std::string &onnx_model_path);
    ~ReIDModel();

    FeatureVector extract_features(cv::Mat &image);

    const std::string &get_distance_metric() const noexcept;

    class InferenceImpl;

private:    
    
    InferenceImpl *_inference_impl = nullptr;
};
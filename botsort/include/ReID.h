#pragma once

#include <opencv2/core.hpp>

#include "DataType.h"
#include "TRT_InferenceEngine/TensorRT_InferenceEngine.h"

class ReIDModel
{
public:
    ReIDModel(const std::string &config_path,
              const std::string &onnx_model_path);
    ~ReIDModel() = default;

    void pre_process(cv::Mat &image);
    FeatureVector extract_features(cv::Mat &image);

    const std::string &get_distance_metric() const
    {
        return _distance_metric;
    }

private:
    void _load_params_from_config(const std::string &config_path);


private:
    inference_backend::TRTOptimizerParams _model_optimization_params;
    std::unique_ptr<inference_backend::TensorRTInferenceEngine>
            _trt_inference_engine;
    u_int8_t _trt_logging_level;
    cv::Size _input_size;

    std::string _onnx_model_path, _distance_metric;
};
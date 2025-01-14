#include "ReID.h"

#include "INIReader.h"

ReIDModel::ReIDModel(const ReIDParams &params,
                     const std::string &onnx_model_path)
{
    std::cout << "Initializing ReID model" << std::endl;
    _load_params_from_config(params);

    _onnx_model_path = onnx_model_path;
    _trt_inference_engine =
            std::make_unique<inference_backend::TensorRTInferenceEngine>(
                    _model_optimization_params, _trt_logging_level);

    bool net_initialized = _trt_inference_engine->load_model(_onnx_model_path);
    if (!net_initialized)
    {
        std::cout << "Failed to initialize ReID model" << std::endl;
        exit(1);
    }
}


FeatureVector ReIDModel::extract_features(cv::Mat &image_patch)
{
    pre_process(image_patch);
    std::vector<std::vector<float>> output =
            _trt_inference_engine->forward(image_patch);

    // TODO: Clean this up
    FeatureVector feature_vector = FeatureVector::Zero(1, FEATURE_DIM);
    for (int i = 0; i < FEATURE_DIM; i++)
        feature_vector(0, i) = output[0][i];

    return feature_vector;
}


void ReIDModel::pre_process(cv::Mat &image)
{
    cv::resize(image, image, _input_size);
    if (_model_optimization_params.swapRB)
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
}

void ReIDModel::_load_params_from_config(const ReIDParams &params)
{
    _distance_metric = params.distance_metric;
    _trt_logging_level = params.trt_logging_level;
    _model_optimization_params.batch_size = static_cast<int>(params.batch_size);
    _model_optimization_params.fp16 = params.enable_fp16;
    _model_optimization_params.tf32 = params.enable_tf32;
    _model_optimization_params.input_layer_name = params.input_layer_name;

    std::cout << "Trying to get input dims" << std::endl;
    const auto &input_dims = params.input_layer_dimensions;
    _input_size = cv::Size(input_dims[3], input_dims[2]);

    std::cout << "Read input dims" << std::endl;
    std::cout << "Input dims: " << input_dims[0] << " " << input_dims[1] << " "
              << input_dims[2] << " " << input_dims[3] << std::endl;

    _model_optimization_params.input_dims = nvinfer1::Dims4{
            input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    _model_optimization_params.swapRB = params.swap_rb;

    _model_optimization_params.output_layer_names = params.output_layer_names;
}
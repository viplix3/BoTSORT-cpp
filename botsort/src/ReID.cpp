#include "ReID.h"

#include "INIReader.h"

ReIDModel::ReIDModel(const std::string &config_path)
{
    _load_params_from_config(config_path);
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
}


void ReIDModel::_load_params_from_config(const std::string &config_path)
{
    const std::string section_name = "ReID";
    INIReader reid_config(config_path);
    if (reid_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    _onnx_model_path = reid_config.Get(section_name, "onnx_model_path", "");
    _distance_metric =
            reid_config.Get(section_name, "distance_metric", "euclidean");
    _trt_logging_level =
            reid_config.GetInteger(section_name, "logging_level", 0);

    _model_optimization_params.batch_size =
            reid_config.GetInteger(section_name, "batch_size", 1);
    _model_optimization_params.fp16 =
            reid_config.GetBoolean(section_name, "fp16", true);
    _model_optimization_params.tf32 =
            reid_config.GetBoolean(section_name, "tf32", true);

    _model_optimization_params.input_layer_name =
            reid_config.Get(section_name, "input_layer_name", "");
    std::vector<int> input_dims =
            reid_config.GetList<int>(section_name, "input_dims");
    _model_optimization_params.input_dims = nvinfer1::Dims4{
            input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    _model_optimization_params.swapRB =
            reid_config.GetBoolean(section_name, "swapRB", false);

    _model_optimization_params.output_layer_names =
            reid_config.GetList<std::string>(section_name,
                                             "output_layer_names");
}
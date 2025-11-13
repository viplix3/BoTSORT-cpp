#include "ReID.h"

#include "INIReader.h"

#ifdef USE_TRT_REID
#include "TRT_InferenceEngine/TensorRT_InferenceEngine.h"
#else
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#endif

///
class ReIDModel::InferenceImpl
{
public:
    InferenceImpl() = default;
    virtual ~InferenceImpl() = default;

    virtual bool Init(const ReIDParams &params, const std::string &onnx_model_path) = 0;

    virtual const std::string &get_distance_metric() const noexcept = 0;
    virtual FeatureVector extract_features(cv::Mat &image_patch) = 0;
};

#ifdef USE_TRT_REID
///
class ReIDInferenceTRT final : public ReIDModel::InferenceImpl
{
public:
    bool Init(const ReIDParams& params, const std::string& onnx_model_path) override
    {
        _onnx_model_path = onnx_model_path;

        _load_params_from_config(params);

        _trt_inference_engine =
                std::make_unique<inference_backend::TensorRTInferenceEngine>(
                        _model_optimization_params, _trt_logging_level);

        auto res = _trt_inference_engine->load_model(onnx_model_path);
        return res;
    }

    FeatureVector extract_features(cv::Mat &image_patch) override
    {
        pre_process(image_patch);
        std::vector<std::vector<float>> output =
                _trt_inference_engine->forward(image_patch);

        // TODO: Clean this up
        FeatureVector feature_vector = FeatureVector::Zero(1, FEATURE_DIM);
        for (int i = 0; i < FEATURE_DIM; i++)
        {
            feature_vector(0, i) = output[0][i];
        }
        return feature_vector;
    }

    const std::string &get_distance_metric() const noexcept override
    {
        return _distance_metric;
    }
    
private:
    inference_backend::TRTOptimizerParams _model_optimization_params;
    std::unique_ptr<inference_backend::TensorRTInferenceEngine> _trt_inference_engine;

    int8_t _trt_logging_level;
    cv::Size _input_size;

    std::string _onnx_model_path;
    std::string _distance_metric;


    void _load_params_from_config(const ReIDParams &params)
    {
        _distance_metric = params.distance_metric;
        _trt_logging_level = params.trt_logging_level;
        _model_optimization_params.batch_size =
                static_cast<int>(params.batch_size);
        _model_optimization_params.fp16 = params.enable_fp16;
        _model_optimization_params.tf32 = params.enable_tf32;
        _model_optimization_params.input_layer_name = params.input_layer_name;

        std::cout << "Trying to get input dims" << std::endl;
        const auto &input_dims = params.input_layer_dimensions;
        _input_size = cv::Size(input_dims[3], input_dims[2]);

        std::cout << "Read input dims" << std::endl;
        std::cout << "Input dims: " << input_dims[0] << " " << input_dims[1]
                  << " " << input_dims[2] << " " << input_dims[3] << std::endl;

        _model_optimization_params.input_dims = nvinfer1::Dims4{
                input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
        _model_optimization_params.swapRB = params.swap_rb;

        _model_optimization_params.output_layer_names =
                params.output_layer_names;
    }

    void pre_process(cv::Mat &image)
    {
        cv::resize(image, image, _input_size);
        if (_model_optimization_params.swapRB)
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }
};

#else // USE_TRT_REID
///
class ReIDInferenceOCV final : public ReIDModel::InferenceImpl
{
public:
    bool Init(const ReIDParams &params,
              const std::string &onnx_model_path) override
    {
        _onnx_model_path = onnx_model_path;

        _load_params_from_config(params);

        _net = cv::dnn::readNet(onnx_model_path);

        std::cout << "Re-id model " << onnx_model_path
                  << " loaded: " << (!_net.empty()) << std::endl;

        if (!_net.empty())
        {
            std::map<cv::dnn::Target, std::string> dictTargets;
            dictTargets[cv::dnn::DNN_TARGET_CPU] = "DNN_TARGET_CPU";
            dictTargets[cv::dnn::DNN_TARGET_OPENCL] = "DNN_TARGET_OPENCL";
            dictTargets[cv::dnn::DNN_TARGET_OPENCL_FP16] =
                    "DNN_TARGET_OPENCL_FP16";
            dictTargets[cv::dnn::DNN_TARGET_MYRIAD] = "DNN_TARGET_MYRIAD";
            dictTargets[cv::dnn::DNN_TARGET_CUDA] = "DNN_TARGET_CUDA";
            dictTargets[cv::dnn::DNN_TARGET_CUDA_FP16] = "DNN_TARGET_CUDA_FP16";
#if (CV_VERSION_MAJOR > 4)
            dictTargets[cv::dnn::DNN_TARGET_HDDL] = "DNN_TARGET_HDDL";
            dictTargets[cv::dnn::DNN_TARGET_NPU] = "DNN_TARGET_NPU";
            dictTargets[cv::dnn::DNN_TARGET_CPU_FP16] = "DNN_TARGET_CPU_FP16";
#endif

            std::map<int, std::string> dictBackends;
            dictBackends[cv::dnn::DNN_BACKEND_DEFAULT] = "DNN_BACKEND_DEFAULT";
            dictBackends[cv::dnn::DNN_BACKEND_INFERENCE_ENGINE] =
                    "DNN_BACKEND_INFERENCE_ENGINE";
            dictBackends[cv::dnn::DNN_BACKEND_OPENCV] = "DNN_BACKEND_OPENCV";
            dictBackends[cv::dnn::DNN_BACKEND_VKCOM] = "DNN_BACKEND_VKCOM";
            dictBackends[cv::dnn::DNN_BACKEND_CUDA] = "DNN_BACKEND_CUDA";
#if (CV_VERSION_MAJOR > 4)
            dictBackends[cv::dnn::DNN_BACKEND_WEBNN] = "DNN_BACKEND_WEBNN";
            dictBackends[cv::dnn::DNN_BACKEND_TIMVX] = "DNN_BACKEND_TIMVX";
            dictBackends[cv::dnn::DNN_BACKEND_CANN] = "DNN_BACKEND_CANN";
#endif
            dictBackends[1000000] = "DNN_BACKEND_INFERENCE_ENGINE_NGRAPH";
            dictBackends[1000000 + 1] =
                    "DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019";

            std::cout << "Avaible pairs for Target - backend:" << std::endl;
            std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> pairs =
                    cv::dnn::getAvailableBackends();
            for (auto p: pairs)
            {
                std::cout << dictBackends[p.first] << " (" << p.first << ") - "
                          << dictTargets[p.second] << " (" << p.second << ")"
                          << std::endl;

                if (p.first == cv::dnn::DNN_BACKEND_CUDA)
                {
                    //_net.setPreferableTarget(p.second);
                    //_net.setPreferableBackend(p.first);
                    //std::cout << "Set!" << std::endl;
                }
            }

            auto outNames = _net.getUnconnectedOutLayersNames();
            auto outLayers = _net.getUnconnectedOutLayers();
            auto outLayerType = _net.getLayer(outLayers[0])->type;

#if (CV_VERSION_MAJOR < 5)
            std::vector<cv::dnn::MatShape> outputs;
            std::vector<cv::dnn::MatShape> internals;
            _net.getLayerShapes(cv::dnn::MatShape(), 0, outputs, internals);
#else
            std::vector<cv::MatShape> outputs;
            std::vector<cv::MatShape> internals;
            _net.getLayerShapes(cv::MatShape(), CV_32F, 0, outputs, internals);
#endif
            std::cout << "REID: getLayerShapes: outputs (" << outputs.size()
                      << ") = " << (outputs.size() > 0 ? outputs[0].size() : 0)
                      << ", internals (" << internals.size() << ") = "
                      << (internals.size() > 0 ? internals[0].size() : 0)
                      << std::endl;
            if (outputs.size() && outputs[0].size() > 3)
                std::cout << "outputs = [" << outputs[0][0] << ", "
                          << outputs[0][1] << ", " << outputs[0][2] << ", "
                          << outputs[0][3] << "], internals = ["
                          << internals[0][0] << ", " << internals[0][1] << ", "
                          << internals[0][2] << ", " << internals[0][3] << "]"
                          << std::endl;
        }
        return !_net.empty();
    }

    FeatureVector extract_features(cv::Mat &image_patch) override
    {
        FeatureVector feature_vector;

        if (!_net.empty())
        {
            cv::Mat obj;
            cv::resize(image_patch, obj, _input_size, 0., 0., cv::INTER_CUBIC);
            cv::Mat blob =
                    cv::dnn::blobFromImage(obj, 1.0 / 255.0, cv::Size(),
                                           cv::Scalar(), _swapRB, false, CV_32F);

            _net.setInput(blob);

            cv::Mat output;
            cv::normalize(_net.forward(), output);

            feature_vector = FeatureVector::Zero(1, FEATURE_DIM);
            for (int i = 0; i < FEATURE_DIM; i++)
            {
                feature_vector(0, i) = output.at<float>(i);
            }
        }
        return feature_vector;
    }

    const std::string &get_distance_metric() const noexcept override
    {
        return _distance_metric;
    }

private:
    cv::dnn::Net _net;

    cv::Size _input_size{128, 256};
    std::string _input_layer_name;
    std::string _onnx_model_path;
    std::string _distance_metric;
    std::vector<std::string> _output_layer_names;
    bool _swapRB = true;

    void _load_params_from_config(const ReIDParams &params)
    {
        _distance_metric = params.distance_metric;
        _input_layer_name = params.input_layer_name;

        std::cout << "Trying to get input dims" << std::endl;
        _input_size = cv::Size(params.input_layer_dimensions[3],
                               params.input_layer_dimensions[2]);

        std::cout << "Read input dims" << std::endl;
        std::cout << "Input dims: " << params.input_layer_dimensions[0] << " "
                  << params.input_layer_dimensions[1] << " "
                  << params.input_layer_dimensions[2] << " "
                  << params.input_layer_dimensions[3] << std::endl;

        _swapRB = params.swap_rb;

        _output_layer_names = params.output_layer_names;
    }
};
#endif // USE_TRT_REID

///
ReIDModel::ReIDModel(const ReIDParams &params,
                     const std::string &onnx_model_path)
{
    std::cout << "Initializing ReID model" << std::endl;

#ifdef USE_TRT_REID
    _inference_impl = new ReIDInferenceTRT();
#else
    _inference_impl = new ReIDInferenceOCV();
#endif

    bool net_initialized = _inference_impl->Init(params, onnx_model_path);
    if (!net_initialized)
    {
        std::cout << "Failed to initialize ReID model" << std::endl;
        exit(1);
    }
}

///
ReIDModel::~ReIDModel()
{
    if (_inference_impl)
        delete _inference_impl;
}

///
FeatureVector ReIDModel::extract_features(cv::Mat &image_patch)
{
    return _inference_impl->extract_features(image_patch);
}

///
const std::string& ReIDModel::get_distance_metric() const noexcept
{
    return _inference_impl->get_distance_metric();
}

#include "TRT_InferenceEngine/TensorRT_InferenceEngine.h"

#include <NvOnnxParser.h>

inference_backend::TensorRTInferenceEngine::TensorRTInferenceEngine(
        TRTOptimizerParams &optimization_params, u_int8_t logging_level)
{
    _set_optimization_params(optimization_params);
    _init_TRT_logger(logging_level);
}


inference_backend::TensorRTInferenceEngine::~TensorRTInferenceEngine()
{
    for (auto &buffer: _buffers)
        cudaFree(buffer);

    cudaStreamDestroy(_cuda_stream);
}


void inference_backend::TensorRTInferenceEngine::_set_optimization_params(
        const TRTOptimizerParams &params)
{
    _optimization_params = params;
}


void inference_backend::TensorRTInferenceEngine::_init_TRT_logger(
        u_int8_t logging_level)
{
    _logger = std::make_unique<TRTLogger>(
            static_cast<nvinfer1::ILogger::Severity>(logging_level));
}


bool inference_backend::TensorRTInferenceEngine::load_model(
        const std::string &onnx_model_path)
{
    _logger->log(nvinfer1::ILogger::Severity::kINFO,
                 std::string("Loading ONNX model from path: ")
                         .append(onnx_model_path)
                         .c_str());

    // Check if ONNX model exists
    if (!file_exists(onnx_model_path))
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("ONNX model not found at path: ")
                             .append(onnx_model_path)
                             .c_str());
        return false;
    }

    // Check if engine exists, if not build it
    std::string engine_path = get_engine_path(onnx_model_path);
    if (!file_exists(engine_path))
    {
        _logger->log(nvinfer1::ILogger::Severity::kINFO,
                     std::string("Engine not found at path: ")
                             .append(engine_path)
                             .c_str());
        _build_engine(onnx_model_path);
    }

    // Deserialize engine
    _logger->log(nvinfer1::ILogger::Severity::kINFO,
                 std::string("Deserializing engine from path: ")
                         .append(engine_path)
                         .c_str());
    if (_deserialize_engine(engine_path))
    {
        allocate_buffers();
        _logger->log(nvinfer1::ILogger::Severity::kINFO,
                     std::string("Engine loaded successfully").c_str());
        return true;
    }

    _logger->log(nvinfer1::ILogger::Severity::kERROR,
                 std::string("Failed to load engine").c_str());
    return false;
}


std::string inference_backend::TensorRTInferenceEngine::get_engine_path(
        const std::string &onnx_model_path) const
{
    // Parent director + model name
    std::string engine_path =
            boost::filesystem::path(onnx_model_path).parent_path().string() +
            "/" + boost::filesystem::path(onnx_model_path).stem().string();

    // Hostname
    char hostname[1024];
    gethostname(hostname, sizeof(hostname));
    std::string suffix(hostname);

    // TensorRT version
    suffix.append("_TRT" + std::to_string(NV_TENSORRT_VERSION));

    // CUDA version
    suffix.append("_CUDA" + std::to_string(CUDART_VERSION));

    // Batch size
    suffix.append("_" + std::to_string(_optimization_params.batch_size));

    // int8, tf32, fp16, fp32
    if (_optimization_params.int8)
        suffix.append("_INT8");
    else
    {
        if (_optimization_params.tf32)
            suffix.append("_TF32");
        _optimization_params.fp16 ? suffix.append("_FP16")
                                  : suffix.append("_FP32");
    }

    // Engine path = parent_dir/model_name_hostname_TRT_version_CUDA_version_batch_size_int8_fp16_fp32.engine
    engine_path.append("_" + suffix + ".engine");
    return engine_path;
}


bool inference_backend::TensorRTInferenceEngine::file_exists(
        const std::string &name) const
{
    return boost::filesystem::exists(name);
}


void inference_backend::TensorRTInferenceEngine::print_engine_info()
{
}


void inference_backend::TensorRTInferenceEngine::allocate_buffers()
{
}


bool inference_backend::TensorRTInferenceEngine::_deserialize_engine(
        const std::string &engine_path)
{
}


void inference_backend::TensorRTInferenceEngine::_build_engine(
        const std::string &onnx_model_path)
{
    // Reference: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#c_topics
    // Network builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(*_logger));

    // Network definition
    uint32_t flag =
            1U << static_cast<uint32_t>(
                    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(flag));

    // ONNX parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, *_logger));

    // Parse ONNX model
    int verbosity = static_cast<int>(_logSeverity);
    parser->parseFromFile(onnx_model_path.c_str(), verbosity);
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     parser->getError(i)->desc());
    }

    // Optimization profile
    // TODO: Check more about optimization profiles
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
            builder->createBuilderConfig());
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(),
                           nvinfer1::OptProfileSelector::kMIN,
                           network->getInput(0)->getDimensions());
    profile->setDimensions(network->getInput(0)->getName(),
                           nvinfer1::OptProfileSelector::kOPT,
                           network->getInput(0)->getDimensions());
    profile->setDimensions(network->getInput(0)->getName(),
                           nvinfer1::OptProfileSelector::kMAX,
                           network->getInput(0)->getDimensions());
    config->addOptimizationProfile(profile);

    if (_optimization_params.int8)

    {
        // TODO: Add int8 calibration
        _logger->log(nvinfer1::ILogger::Severity::kWARNING,
                     std::string("INT8 calibration is not supported yet. "
                                 "Switching to FP16 or FP32 calibration")
                             .c_str());
    }
    else
    {
        _logger->log(nvinfer1::ILogger::Severity::kINFO,
                     std::string("FP16 or FP32 calibration").c_str());
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        if (_optimization_params.tf32)
            config->setFlag(nvinfer1::BuilderFlag::kTF32);
    }

    // Build engine
    _logger->log(nvinfer1::ILogger::Severity::kINFO,
                 std::string("Building engine").c_str());

    std::unique_ptr<nvinfer1::IHostMemory> engine_plan{
            builder->buildSerializedNetwork(*network, *config)};
    std::unique_ptr<nvinfer1::IRuntime> runtime{
            nvinfer1::createInferRuntime(*_logger)};
    std::shared_ptr<nvinfer1::ICudaEngine> engine =
            std::shared_ptr<nvinfer1::ICudaEngine>(
                    runtime->deserializeCudaEngine(engine_plan->data(),
                                                   engine_plan->size()),
                    TRTDestroyer());
    if (!engine)
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Failed to build engine").c_str());
        return;
    }

    // Serialize engine
    _logger->log(nvinfer1::ILogger::Severity::kINFO,
                 std::string("Serializing engine").c_str());
    std::string engine_path = get_engine_path(onnx_model_path);
    std::ofstream engine_file(engine_path, std::ios::binary);
    if (!engine_file)
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Failed to open engine file").c_str());
        return;
    }

    std::unique_ptr<nvinfer1::IHostMemory> serialized_engine{
            engine->serialize()};
    if (!serialized_engine)
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Failed to serialize engine").c_str());
        return;
    }

    engine_file.write(reinterpret_cast<const char *>(serialized_engine->data()),
                      serialized_engine->size());
    if (engine_file.fail())
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Failed to write engine file").c_str());
        return;
    }

    _logger->log(nvinfer1::ILogger::Severity::kINFO,
                 std::string("Engine serialized successfully").c_str());
    engine_file.close();
}


inference_backend::ModelPredictions
inference_backend::TensorRTInferenceEngine::forward(const cv::Mat &input_image)
{
}
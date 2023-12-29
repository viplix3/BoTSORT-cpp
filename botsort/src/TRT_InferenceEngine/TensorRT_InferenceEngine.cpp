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
        _allocate_buffers();
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


void inference_backend::TensorRTInferenceEngine::_allocate_buffers()
{
    for (void *&buffer: _buffers)
        cudaFree(buffer);
    _buffers.clear();

#if NVINFER_MAJOR == 8 && NVINFER_MINOR <= 5
    _buffers = std::vector<void *>(_engine->getNbBindings());
#else
    _buffers = std::vector<void *>(_engine->getNbIOTensors());
#endif
    size_t output_idx = 0;

#if NVINFER_MAJOR == 8 && NVINFER_MINOR <= 5
    for (size_t i = 0; i < _engine->getNbBindings(); ++i)
#else
    for (size_t i = 0; i < _engine->getNbIOTensors(); ++i)
#endif
    {
#if NVINFER_MAJOR == 8 && NVINFER_MINOR <= 5
        nvinfer1::Dims dims = _engine->getBindingDimensions(i);
#else
        const char *name = _engine->getIOTensorName(i);
        nvinfer1::Dims dims = _engine->getTensorShape(name);
#endif

#if NVINFER_MAJOR == 8 && NVINFER_MINOR <= 5
        nvinfer1::DataType dtype = _engine->getBindingDataType(i);
#else
        nvinfer1::DataType dtype = _engine->getTensorDataType(name);
#endif

        size_t total_size = std::accumulate(dims.d, dims.d + dims.nbDims, 1,
                                            std::multiplies<size_t>());
        cudaMalloc(&_buffers[i], total_size * sizeof(float));

#if NVINFER_MAJOR == 8 && NVINFER_MINOR <= 5
        if (_engine->getBindingName(i) == _optimization_params.input_layer_name)
#else
        if (std::string(name) == _optimization_params.input_layer_name)
#endif
        {
            _input_dims.emplace_back(dims);
            _input_idx = i;

#if NVINFER_MAJOR == 8 && NVINFER_MINOR <= 5
            _logger->log(nvinfer1::ILogger::Severity::kINFO,
                         std::string("Found input layer with name: ")
                                 .append(_engine->getBindingName(i))
                                 .c_str());
#else
            _logger->log(nvinfer1::ILogger::Severity::kINFO,
                         std::string("Found input layer with name: ")
                                 .append(name)
                                 .c_str());
#endif
        }

#if NVINFER_MAJOR == 8 && NVINFER_MINOR <= 5
        else if (std::find(_optimization_params.output_layer_names.begin(),
                           _optimization_params.output_layer_names.end(),
                           _engine->getBindingName(i)) !=
                 _optimization_params.output_layer_names.end())
#else
        else if (std::find(_optimization_params.output_layer_names.begin(),
                           _optimization_params.output_layer_names.end(),
                           name) !=
                 _optimization_params.output_layer_names.end())
#endif
        {
            _output_dims.emplace_back(dims);
            _output_idx.emplace_back(i);
            ++output_idx;

#if NVINFER_MAJOR == 8 && NVINFER_MINOR <= 5
            _logger->log(nvinfer1::ILogger::Severity::kINFO,
                         std::string("Found output layer with name: ")
                                 .append(_engine->getBindingName(i))
                                 .c_str());
#else
            _logger->log(nvinfer1::ILogger::Severity::kINFO,
                         std::string("Found output layer with name: ")
                                 .append(name)
                                 .c_str());
#endif
        }
    }

    if (_input_dims.empty())
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Input layer not found").c_str());
        return;
    }
    if (_output_dims.empty())
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Output layer not found").c_str());
        return;
    }
}


bool inference_backend::TensorRTInferenceEngine::_deserialize_engine(
        const std::string &engine_path)
{
    // Reference: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#perform_inference_c
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file)
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Failed to open engine file").c_str());
        return false;
    }

    // Read engine file
    std::vector<char> trt_model_stream;
    size_t size{0};
    engine_file.seekg(0, std::ios::end);
    size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    trt_model_stream.resize(size);
    engine_file.read(trt_model_stream.data(), size);
    engine_file.close();

    // Deserialize engine
    std::unique_ptr<nvinfer1::IRuntime> runtime{
            nvinfer1::createInferRuntime(*_logger)};
    _engine = makeUnique(runtime->deserializeCudaEngine(
            trt_model_stream.data(), trt_model_stream.size()));
    if (!_engine)
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Failed to deserialize engine").c_str());
        return false;
    }

    // Create execution context
    _context = makeUnique(_engine->createExecutionContext());
    if (!_context)
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Failed to create execution context").c_str());
        return false;
    }

    return true;
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
    // Reference: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#perform-inference

    if (!_buffers.size())
    {
        _logger->log(nvinfer1::ILogger::Severity::kERROR,
                     std::string("Buffers not allocated").c_str());
        return ModelPredictions();
    }

    // Create a blob from the image
    cv::Mat image_blob;
    cv::dnn::blobFromImage(input_image, image_blob, 1.0,
                           cv::Size(_input_dims[0].d[3], _input_dims[0].d[2]),
                           cv::Scalar(), false, false, CV_32F);

    // Ensure image blob size matches input layer size
    assert(image_blob.total() ==
           std::accumulate(_input_dims[0].d,
                           _input_dims[0].d + _input_dims[0].nbDims, 1,
                           std::multiplies<size_t>()));

    // Copy image blob to CUDA input buffer
    cudaMemcpyAsync(_buffers[_input_idx], image_blob.ptr<float>(),
                    image_blob.total() * sizeof(float), cudaMemcpyHostToDevice);

    // Run inference
    _context->enqueueV2(_buffers.data(), _cuda_stream, nullptr);

    // Copy CUDA output buffer to host
    ModelPredictions predictions;
    for (size_t i = 0; i < _output_idx.size(); ++i)
    {
        std::vector<float> output(_output_dims[i].d[0] * _output_dims[i].d[1] *
                                  _output_dims[i].d[2] * _output_dims[i].d[3]);
        cudaMemcpyAsync(output.data(), _buffers[_output_idx[i]],
                        output.size() * sizeof(float), cudaMemcpyDeviceToHost);
        predictions.emplace_back(output);
    }

    return predictions;
}
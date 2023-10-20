#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include "NvInfer.h"

using ModelPredictions = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

static auto StreamDeleter = [](cudaStream_t *ptr) {
    if (ptr)
    {
        cudaStreamDestroy(*ptr);
        delete ptr;
    }
};

struct TRTDeleter
{
    template<typename T>
    void operator()(T *obj) const
    {
        if (obj) { obj->destroy(); }
    }
};

struct TRTDestroyer
{
    template<typename T>
    void operator()(T *obj) const
    {
        if (obj) { obj->destroy(); }
    }
};

template<typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

template<typename T>
inline TRTUniquePtr<T> infer_object(T *obj)
{
    if (!obj) { throw std::runtime_error("Failed to create object"); }
    return TRTUniquePtr<T>(obj);
}

struct TRTOptimizerParams
{
    TRTOptimizerParams() = default;

    int batchSize = 1;
    size_t maxWorkspaceSize = 1 << 30;
    bool fp16 = true;
    bool int8 = false;
    bool int8Calibrator = false;
    std::string enginePath = "";
    std::string calibTablePath = "";
    std::string calibCachePath = "";
    int calibBatchSize = 1;
    int calibMaxBatches = 100;
    int calibMaxCachedBatches = 50;
    bool calibUseEntropy = false;
    bool calibUseAvgTiming = false;
    std::string calibCacheFile = "";
    int calibCacheMode = 0;
    bool calibCacheDebug = false;
    bool calibCacheSkip = false;
    bool calibCacheCalibrate = false;
    bool calibCacheCalibrateOnly = false;
    bool calibCacheEvaluate = false;
    bool calibCacheEvaluateOnly = false;
    bool calibCacheVerbose = false;

    std::string toStr()
    {
        std::string str;
        str += "batchSize: " + std::to_string(batchSize) + "\n";
        str += "maxWorkspaceSize: " + std::to_string(maxWorkspaceSize) + "\n";
        str += "fp16: " + std::to_string(fp16) + "\n";
        str += "int8: " + std::to_string(int8) + "\n";
        str += "int8Calibrator: " + std::to_string(int8Calibrator) + "\n";
        str += "enginePath: " + enginePath + "\n";
        str += "calibTablePath: " + calibTablePath + "\n";
        str += "calibCachePath: " + calibCachePath + "\n";
        str += "calibBatchSize: " + std::to_string(calibBatchSize) + "\n";
        str += "calibMaxBatches: " + std::to_string(calibMaxBatches) + "\n";
        str += "calibMaxCachedBatches: " +
               std::to_string(calibMaxCachedBatches) + "\n";
        str += "calibUseEntropy: " + std::to_string(calibUseEntropy) + "\n";
        str += "calibUseAvgTiming: " + std::to_string(calibUseAvgTiming) + "\n";
        str += "calibCacheFile: " + calibCacheFile + "\n";
        str += "calibCacheMode: " + std::to_string(calibCacheMode) + "\n";
        str += "calibCacheDebug: " + std::to_string(calibCacheDebug) + "\n";
        str += "calibCacheSkip: " + std::to_string(calibCacheSkip) + "\n";
        str += "calibCacheCalibrate: " + std::to_string(calibCacheCalibrate) +
               "\n";
        str += "calibCacheCalibrateOnly: " +
               std::to_string(calibCacheCalibrateOnly) + "\n";
        str += "calibCacheEvaluate: " + std::to_string(calibCacheEvaluate) +
               "\n";
        str += "calibCacheEvaluateOnly: " +
               std::to_string(calibCacheEvaluateOnly) + "\n";
        str += "calibCacheVerbose: " + std::to_string(calibCacheVerbose) + "\n";
        return str;
    }
};


class TensortRTInferenceEngine
{
private:
    nvinfer1::ILogger::Severity _logSeverity =
            nvinfer1::ILogger::Severity::kWARNING;
    TRTOptimizerParams _optimization_params;
    TRTUniquePtr<nvinfer1::ICudaEngine> _engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> _context{nullptr};

    cudaStream_t _cuda_stream;

    std::vector<void *> _bindings;
    std::vector<nvinfer1::Dims> _input_dims;
    std::vector<nvinfer1::Dims> _output_dims;
    std::vector<std::string> _output_layer_names;

    int input_idx = 0;
    std::vector<int> output_idx;

public:
    TensortRTInferenceEngine();
    ~TensortRTInferenceEngine();

    void setOptimizationParams(const TRTOptimizerParams &params);

    void setSeverityLevel(const nvinfer1::ILogger::Severity &severity);
    void setTensorRTLoggingLevel(int level);

    bool loadModel(const std::string &modelPath);
    std::string getEnginePath(const std::string &onnxModelPath);

    void buildEngine(const std::string &onnxModelPath);
    ModelPredictions forward(const cv::Mat &input_image);

    void printEngineInfo();
    void allocateBuffers();
};


class TRTLogger : public nvinfer1::ILogger
{
public:
    TRTLogger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};
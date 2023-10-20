#pragma once

#include <opencv2/core.hpp>

#include "DataType.h"

class ReIDModel
{
public:
    ReIDModel(std::string model_weights, bool fp16_inference);
    ~ReIDModel() = default;

    FeatureVector extract_features(cv::Mat &image_patch);
};
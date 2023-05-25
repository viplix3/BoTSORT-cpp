#pragma once

#include "DataType.h"

#include <opencv2/core.hpp>

class ReIDModel {
public:
    ReIDModel(std::string model_weights, bool fp16_inference);
    ~ReIDModel() = default;

    FeatureVector extract_features(cv::Mat &image_patch);
};
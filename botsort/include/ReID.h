#pragma once

#include <opencv2/core.hpp>

#include "DataType.h"

class ReIDModel
{
public:
    ReIDModel(const std::string &config_path);
    ~ReIDModel() = default;

    FeatureVector extract_features(cv::Mat &image_patch);
};
#include "ReID.h"

ReIDModel::ReIDModel(const char *model_weights, bool fp16_inference) {
}

FeatureVector ReIDModel::extract_features(cv::Mat &image_patch) {
    return FeatureVector();
}
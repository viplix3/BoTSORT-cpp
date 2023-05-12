#pragma once

#include "DataType.h"

/**
 * @brief Calculate the cosine distance between two feature vectors
 * 
 * @param x Feature vector 1
 * @param y Feature vector 2
 * @return float Cosine distance (1 - cosine similarity)
 */
inline float cosine_distance(const FeatureVector &x, const FeatureVector &y);

/**
 * @brief Calculate the intersection over union (IoU) between two bounding boxes
 * 
 * @param tlwh_a Bounding box 1 in the format (top left x, top left y, width, height)
 * @param tlwh_b Bounding box 2 in the format (top left x, top left y, width, height)
 * @return float IoU
 */
inline float iou(const std::vector<float> &tlwh_a, const std::vector<float> &tlwh_b);

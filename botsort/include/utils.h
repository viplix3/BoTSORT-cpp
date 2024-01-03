#pragma once

#include "DataType.h"

/**
 * @brief Calculate the cosine distance between two feature vectors
 * 
 * @param x Feature vector 1
 * @param y Feature vector 2
 * @return float Cosine distance (1 - cosine similarity)
 */
inline float cosine_distance(const std::unique_ptr<FeatureVector> &x,
                             const std::shared_ptr<FeatureVector> &y)
{
    return 1.0f - (x->dot(*y) / (x->norm() * y->norm() + 1e-5f));
}


/**
 * @brief Calculate the euclidean distance between two feature vectors
 * 
 * @param x Feature vector 1
 * @param y Feature vector 2
 * @return float Euclidean distance
 */
inline float euclidean_distance(const std::unique_ptr<FeatureVector> &x,
                                const std::shared_ptr<FeatureVector> &y)
{
    return (x->transpose() - y->transpose()).norm();
}


/**
 * @brief Calculate the intersection over union (IoU) between two bounding boxes
 * 
 * @param tlwh_a Bounding box 1 in the format (top left x, top left y, width, height)
 * @param tlwh_b Bounding box 2 in the format (top left x, top left y, width, height)
 * @return float IoU
 */
inline float iou(const std::vector<float> &tlwh_a,
                 const std::vector<float> &tlwh_b)
{
    float left = std::max(tlwh_a[0], tlwh_b[0]);
    float top = std::max(tlwh_a[1], tlwh_b[1]);
    float right = std::min(tlwh_a[0] + tlwh_a[2], tlwh_b[0] + tlwh_b[2]);
    float bottom = std::min(tlwh_a[1] + tlwh_a[3], tlwh_b[1] + tlwh_b[3]);
    float area_i =
            std::max(right - left + 1, 0.0f) * std::max(bottom - top + 1, 0.0f);
    float area_a = (tlwh_a[2] + 1) * (tlwh_a[3] + 1);
    float area_b = (tlwh_b[2] + 1) * (tlwh_b[3] + 1);
    return area_i / (area_a + area_b - area_i);
}

double lapjv(CostMatrix &cost, std::vector<int> &rowsol,
             std::vector<int> &colsol, bool extend_cost = false,
             float cost_limit = std::numeric_limits<float>::max(),
             bool return_cost = true);
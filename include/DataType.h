#pragma once

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <utility>
#include <vector>

constexpr uint8_t DET_ELEMENTS = 4;
constexpr uint8_t FEATURE_DIM = 128;
constexpr uint8_t KALMAN_STATE_SPACE_DIM = 8;
constexpr uint8_t KALMAN_MEASUREMENT_SPACE_DIM = 4;

// Detection
/**
 * @brief Detection vector with DET_ELEMENTS elements.
 */
using DetVec = Eigen::Matrix<float, 1, DET_ELEMENTS>;
/**
 * @brief Detection matrix with dynamic rows and DET_ELEMENTS columns.
 */
using DetMatrix = Eigen::Matrix<float, Eigen::Dynamic, DET_ELEMENTS>;

// Re-ID Features
/**
 * @brief Re-ID feature vector with FEATURE_DIM elements.
 */
using FeatureVector = Eigen::Matrix<float, 1, FEATURE_DIM>;
/**
 * @brief Re-ID feature matrix with dynamic rows and FEATURE_DIM columns.
 */
using FeatureMatrix = Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIM>;

// Kalman Filter
/**
 * @brief Kalman Filter state space vector with KALMAN_STATE_SPACE_DIM elements.
 */
using KFStateSpaceVec = Eigen::Matrix<float, 1, KALMAN_STATE_SPACE_DIM>;
/**
 * @brief Kalman Filter state space matrix with KALMAN_STATE_SPACE_DIM rows and columns.
 */
using KFStateSpaceMatrix = Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM>;
/**
 * @brief Kalman Filter state space data containing a mean vector and a covariance matrix.
 */
using KFDataStateSpace = std::pair<KFStateSpaceVec, KFStateSpaceMatrix>;

/**
 * @brief Kalman Filter measurement space vector with KALMAN_MEASUREMENT_SPACE_DIM elements.
 */
using KFMeasSpaceVec = Eigen::Matrix<float, 1, KALMAN_MEASUREMENT_SPACE_DIM, Eigen::RowMajor>;
/**
 * @brief Kalman Filter measurement space matrix with KALMAN_MEASUREMENT_SPACE_DIM rows and columns.
 */
using KFMeasSpaceMatrix = Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_MEASUREMENT_SPACE_DIM, Eigen::RowMajor>;
/**
 * @brief Kalman Filter measurement space data containing a mean vector and a covariance matrix.
 */
using KFDataMeasurementSpace = std::pair<KFMeasSpaceVec, KFMeasSpaceMatrix>;

// Tracker
/**
 * @brief Tracker data containing a track ID and a feature vector.
 */
using TrackerData = std::pair<int, FeatureVector>;
/**
 * @brief Match data containing a track ID and a detection ID.
 */
using MatchData = std::pair<int, int>;

/**
 * @brief Association data containing matched and unmatched tracks and detections.
 */
struct AssociationData {
    std::vector<MatchData> matches;       ///< Matched track and detection pairs.
    std::vector<int> unmatched_tracks;    ///< Unmatched track IDs.
    std::vector<int> unmatched_detections;///< Unmatched detection IDs.
};

// Linear Assignment
/**
 * @brief Cost matrix for linear assignment with dynamic rows and columns.
 */
using CostMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Result
/**
 * @brief Tracking result data containing a track ID and a detection vector.
 */
using ResultData = std::pair<int, DetVec>;

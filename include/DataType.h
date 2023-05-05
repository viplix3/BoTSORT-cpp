#pragma once

#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

constexpr uint8_t DET_ELEMENTS = 4;
constexpr uint8_t FEATURE_DIM = 128;
constexpr uint8_t KALMAN_STATE_SPACE_DIM = 8;
constexpr uint8_t KALMAN_MEASUREMENT_SPACE_DIM = 4;


typedef Eigen::Matrix<float, 1, DET_ELEMENTS> DET_VEC;
typedef Eigen::Matrix<float, Eigen::Dynamic, DET_ELEMENTS> DET_MATRIX;


// Re-ID features
/**
 * @brief Re-ID feature vector
 * 
 */
typedef Eigen::Matrix<float, 1, FEATURE_DIM> FEATURE_VECTOR;
/**
 * @brief Re-ID feature vector matrix [num_features, FEATURE_DIM]
 * 
 */
typedef Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIM> FEATURE_MATRIX;


// Kalman Filter
/**
 * @brief 1xKALMAX_STATE_SPACE_DIM vector
 * 
 */
typedef Eigen::Matrix<float, 1, KALMAN_STATE_SPACE_DIM> KF_STATE_SPACE_VEC;
/**
 * @brief KALMAN_STATE_SPACE_DIMxKALMAN_STATE_SPACE_DIM matrix
 * 
 */
typedef Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM> KF_STATE_SPACE_MATRIX;
/**
 * @brief Kalman filter state space data [mean, covariance]
 * 
 */
using KF_DATA_STATE_SPACE = std::pair<KF_STATE_SPACE_VEC, KF_STATE_SPACE_MATRIX>;

/**
 * @brief 1xKALMAN_MEASUREMENT_SPACE_DIM vector
 * 
 */
typedef Eigen::Matrix<float, 1, KALMAN_MEASUREMENT_SPACE_DIM, Eigen::RowMajor> KF_MEAS_SPACE_VEC;
/**
 * @brief KALMAN_MEASUREMENT_SPACE_DIMxKALMAN_MEASUREMENT_SPACE_DIM matrix
 * 
 */
typedef Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_MEASUREMENT_SPACE_DIM, Eigen::RowMajor> KF_MEAS_SPACE_MATRIX;
/**
 * @brief Kalman filter measurement space data [mean, covariance]
 * 
 */
using KF_DATA_MEASUREMENT_SPACE = std::pair<KF_MEAS_SPACE_VEC, KF_MEAS_SPACE_MATRIX>;


// Tracker
/**
 * @brief Tracker data [track_id, feature_vector]
 * 
 */
using TRACKER_DATA = std::pair<int, FEATURE_VECTOR>;
/**
 * @brief Match data [track_id, detection_id]
 * 
 */
using MATCH_DATA = std::pair<int, int>;
/**
 * @brief Association data [matches, unmatched_tracks, unmatched_detections]
 * 
 */
typedef struct {
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
} ASSOCIATION_DATA;


// Linear Assignment
/**
 * @brief Cost matrix for linear assignment [num_tracks, num_detections]
 * 
 */
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> COST_MATRIX;


// Result
/**
 * @brief Tracking output [track_id, detection]
 * 
 */
using RESULT_DATA = std::pair<int, DET_VEC>;
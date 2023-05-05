#pragma once

#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

constexpr uint8_t DET_ELEMENTS = 6;
constexpr uint8_t FEATURE_DIM = 128;
constexpr uint8_t KALMAN_STATE_SPACE_DIM = 8;
constexpr uint8_t KALMAN_MEASUREMENT_SPACE_DIM = 4;


// Object Detection
/**
 * @brief Detection [x1, y1, x2, y2, class_id, score]
 * 
 */
typedef Eigen::Matrix<float, 1, DET_ELEMENTS, Eigen::RowMajor> DET;
/**
 * @brief Detection matrix [num_detections, DET_ELEMENTS]
 * 
 */
typedef Eigen::Matrix<float, Eigen::Dynamic, DET_ELEMENTS, Eigen::RowMajor> DET_MATRIX;


// Re-ID features
/**
 * @brief Re-ID feature vector
 * 
 */
typedef Eigen::Matrix<float, 1, FEATURE_DIM, Eigen::RowMajor> FEATURE_VECTOR;
/**
 * @brief Re-ID feature vector matrix [num_features, FEATURE_DIM]
 * 
 */
typedef Eigen::Matrix<float, Eigen::Dynamic, FEATURE_DIM, Eigen::RowMajor> FEATURE_MATRIX;


// Kalman Filter
/**
 * @brief Kalman filter mean vector (state space)
 *  BoT-SORT uses 8 dimensional state space [x, y, w, h, vx, vy, vw, vh]
 * 
 */
typedef Eigen::Matrix<float, 1, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> KF_MEAN_STATE_SPACE;
/**
 * @brief Kalman filter covariance matrix (state space)
 * 
 */
typedef Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> KF_COVARIANCE_STATE_SPACE;
/**
 * @brief Kalman filter state space data [mean, covariance]
 * 
 */
using KF_DATA_STATE_SPACE = std::pair<KF_MEAN_STATE_SPACE, KF_COVARIANCE_STATE_SPACE>;

/**
 * @brief Kalman filter mean vector (measurement space).
 *  In BoT-SORT, measurement space is 4 dimensional [x, y, w, h]
 * 
 */
typedef Eigen::Matrix<float, 1, KALMAN_MEASUREMENT_SPACE_DIM, Eigen::RowMajor> KF_MEAN_MEASUREMENT_SPACE;
/**
 * @brief Kalman filter covariance matrix (measurement space)
 * 
 */
typedef Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_MEASUREMENT_SPACE_DIM, Eigen::RowMajor> KF_COVARIANCE_MEASUREMENT_SPACE;
/**
 * @brief Kalman filter measurement space data [mean, covariance]
 * 
 */
using KF_DATA_MEASUREMENT_SPACE = std::pair<KF_MEAN_MEASUREMENT_SPACE, KF_COVARIANCE_MEASUREMENT_SPACE>;


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
using RESULT_DATA = std::pair<int, DET>;
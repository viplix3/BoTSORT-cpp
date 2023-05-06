#pragma once

#include "DataType.h"

namespace kalman_modified {
class KalmanFilter {
public:
    static const double chi2inv95[10];

    /**
     * @brief Construct a new Kalman Filter object
     * 
     * @param dt Time interval between consecutive measurements (dt = 1/FPS). Defaults to 1.0
     */
    KalmanFilter(double dt);

    /**
     * @brief Initialize Kalman Filter with measurement (detections)
     * 
     * @param det Detection [x, y, w, h]
     * @return KFDataStateSpace Kalman filter state space data [mean, covariance]
     */
    KFDataStateSpace init(const DetVec &det);

    void predict(KFStateSpaceVec &mean, KFStateSpaceMatrix &covariance);
    KFDataMeasurementSpace KalmanFilter::project(const KFStateSpaceVec &mean, const KFStateSpaceMatrix &covariance, bool motion_compensated = false);
    KFDataStateSpace update(const KFStateSpaceVec &mean, const KFStateSpaceMatrix &covariance, const DetVec &measurement);

    Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance(
            const KFStateSpaceVec &mean,
            const KFStateSpaceMatrix &covariance,
            const std::vector<DetVec> &measurements);

private:
    /**
     * @brief Initialize Kalman Filter matrices (state transition, measurement, process noise covariance)
     * 
     * @param dt Time interval between consecutive measurements (dt = 1/FPS)
     */
    void _init_kf_matrices(double dt);

    float _init_pos_weight, _init_vel_weight;
    float _std_factor_acceleration, _std_offset_acceleration;
    float _std_factor_detection, _min_std_detection;
    float _std_factor_motion_compensated_detection, _min_std_motion_compensated_detection;
    float _velocity_coupling_factor;
    uint8_t _velocity_half_life;

    Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _state_transition_matrix;
    Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _measurement_matrix;
    Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _process_noise_covariance;
};
}// namespace kalman_modified

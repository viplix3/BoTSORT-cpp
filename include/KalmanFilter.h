#pragma once

#include "DataType.h"

namespace byte_kalman {
class KalmanFilter {
public:
    static const double chi2inv95[10];

    /**
     * @brief Construct a new Kalman Filter object.
     * 
     * @param dt Time interval between consecutive measurements (dt = 1/FPS). Defaults to 1.0.
     */
    KalmanFilter(double dt);

    /**
     * @brief Initialize the Kalman Filter with a measurement (detection).
     * 
     * @param det Detection [x, y, w, h].
     * @return KFDataStateSpace Kalman filter state space data [mean, covariance].
     */
    KFDataStateSpace init(const DetVec &det);

    void predict(KFStateSpaceVec &mean, KFStateSpaceMatrix &covariance);
    KFDataMeasurementSpace project(const KFStateSpaceVec &mean, const KFStateSpaceMatrix &covariance);
    KFDataStateSpace update(const KFDataStateSpace &state, const DetVec &measurement);

    Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance(
            const KFStateSpaceVec &mean,
            const KFStateSpaceMatrix &covariance,
            const std::vector<DetVec> &measurements,
            bool only_position = false);

private:
    /**
     * @brief Initialize Kalman Filter matrices (state transition, measurement, process noise covariance).
     * 
     * @param dt Time interval between consecutive measurements (dt = 1/FPS).
     */
    void _init_kf_matrices(double dt);

    float _std_weight_position, _std_weight_velocity;

    Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _state_transition_matrix;
    Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _measurement_matrix;
};
}// namespace byte_kalman

#pragma once

#include "DataType.h"

namespace byte_kalman {
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
     * @return KF_DATA_MEASUREMENT_SPACE Kalman filter state space data [mean, covariance]
     */
    KF_DATA_MEASUREMENT_SPACE init(const DET_VEC &det);

    void predict(KF_STATE_SPACE_VEC &mean, KF_STATE_SPACE_MATRIX &covariance);
    KF_DATA_MEASUREMENT_SPACE project(const KF_DATA_STATE_SPACE &state);
    KF_DATA_STATE_SPACE update(const KF_DATA_STATE_SPACE &state, const DET_VEC &measurement);

    Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance(
            const KF_STATE_SPACE_VEC &mean,
            const KF_STATE_SPACE_MATRIX &covariance,
            const std::vector<DET_VEC> &measurements,
            bool only_position = false);

private:
    /**
     * @brief Initialize Kalman Filter matrices (state transition, measurement, process noise covariance)
     * 
     * @param dt Time interval between consecutive measurements (dt = 1/FPS)
     */
    void _init_kf_matrices(double dt);

    float _std_weight_position, _std_weight_velocity;

    Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _state_transition_matrix;
    Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _measurement_matrix;
};
}// namespace byte_kalman

#pragma once

#include "DataType.h"

namespace byte_kalman {
class KalmanFilter {
public:
    static const double chi2inv95[10];
    KalmanFilter();
    KF_DATA_MEASUREMENT_SPACE init(const DET &det);
    void predict(KF_DATA_STATE_SPACE &state);
    KF_DATA_MEASUREMENT_SPACE project(const KF_DATA_STATE_SPACE &state);
    KF_DATA_STATE_SPACE update(const KF_DATA_STATE_SPACE &state, const DET &measurement);

    Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance(
            const KF_MEAN_STATE_SPACE &mean,
            const KF_COVARIANCE_STATE_SPACE &covariance,
            const std::vector<DET> &measurements,
            bool only_position = false);

private:
    Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _motion_matrix;
    Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM, Eigen::RowMajor> _update_matrix;
    float _std_weight_position;
    float _std_weight_velocity;
};
}// namespace byte_kalman
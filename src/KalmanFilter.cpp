#include "KalmanFilter.h"
#include <eigen3/Eigen/Cholesky>

namespace byte_kalman {
const double KalmanFilter::chi2inv95[10] = {
        0,
        3.8415,
        5.9915,
        7.8147,
        9.4877,
        11.070,
        12.592,
        14.067,
        15.507,
        16.919};

KalmanFilter::KalmanFilter(double dt = 1.0) {
    _init_kf_matrices(dt);

    _std_weight_position = 1.0 / 20;
    _std_weight_velocity = 1.0 / 160;
}

void KalmanFilter::_init_kf_matrices(double dt) {
    _measurement_matrix = Eigen::MatrixXf::Identity(KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM);

    _state_transition_matrix = Eigen::MatrixXf::Identity(KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM);
    for (uint8_t i = 0; i < 4; i++) {
        _state_transition_matrix(i, i + 4) = dt;
    }
}

KFDataStateSpace KalmanFilter::init(const DetVec &measurement) {
    constexpr float init_velocity = 0.0;
    KFStateSpaceVec mean_state_space;

    for (uint8_t i = 0; i < KALMAN_STATE_SPACE_DIM; i++) {
        if (i < 4) {
            mean_state_space(i) = measurement(i);
        } else {
            mean_state_space(i) = init_velocity;
        }
    }

    float w = measurement(2), h = measurement(3);
    KFStateSpaceVec std;
    for (uint8_t i = 0; i < KALMAN_STATE_SPACE_DIM; i++) {
        if (i < 4) {
            std(i) = 2 * _std_weight_position * (i % 2 == 0 ? w : h);
        } else {
            std(i) = 10 * _std_weight_velocity * (i % 2 == 0 ? w : h);
        }
    }
    KFStateSpaceVec std_squared = std.array().square();
    KFStateSpaceMatrix covariance = std_squared.asDiagonal();
    return std::make_pair(mean_state_space, covariance);
}

void KalmanFilter::predict(KFStateSpaceVec &mean, KFStateSpaceMatrix &covariance) {
}

}// namespace byte_kalman

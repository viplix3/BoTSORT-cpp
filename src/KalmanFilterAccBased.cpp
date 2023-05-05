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

    _init_pos_weight = 5.0;
    _init_vel_weight = 15.0;
    _std_factor_acceleration = 50.25;
    _std_offset_acceleration = 100.5;
    _std_factor_detection = 0.10;
    _min_std_detection = 4.0;
    _velocity_coupling_factor = 0.6;
    _velocity_half_life = 2;
}

void KalmanFilter::_init_kf_matrices(double dt) {
    /**
     * this is a 4x8 matrix that maps the 8-dimensional state space vector []
     * to the 4-dimensional measurement space vector []
     */
    _measurement_matrix = Eigen::MatrixXf::Identity(KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM);


    /**
     * this is a 8x8 matrix that defines the state transition function
     * it maps the current state space vector to the next state space vector
     */
    _state_transition_matrix = Eigen::MatrixXf::Identity(KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM);
    for (uint8_t i = 0; i < 4; i++) {
        _state_transition_matrix(i, i + 4) = _velocity_coupling_factor * dt;
        _state_transition_matrix(i, (i + 2) % 4 + 4) = (1.0 - _velocity_coupling_factor) * dt;
        _state_transition_matrix(i + 4, i + 4) = std::pow(0.5, (dt / _velocity_half_life));
    }

    /**
     * this is a 8x8 matrix that defines the process noise covariance matrix
     * this takes into account acceleration and jerk for modeling the process noise
     */
    _process_noise_covariance = Eigen::MatrixXf::Identity(KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM);
    for (uint8_t i = 0; i < 4; i++) {
        _process_noise_covariance(i, i) = std::pow(dt, 4) / 4 + std::pow(dt, 2);
        _process_noise_covariance(i, i + 4) = std::pow(dt, 3) / 2;
        _process_noise_covariance(i + 4, i) = std::pow(dt, 3) / 2;
        _process_noise_covariance(i + 4, i + 4) = std::pow(dt, 2);
    }
}

KF_DATA_MEASUREMENT_SPACE KalmanFilter::init(const DET_VEC &measurement) {
    constexpr float init_velocity = 0.0;
    KF_STATE_SPACE_VEC mean_state_space;

    for (uint8_t i = 0; i < KALMAN_STATE_SPACE_DIM; i++) {
        if (i < 4) {
            mean_state_space(i) = measurement(i);
        } else {
            mean_state_space(i) = init_velocity;
        }
    }

    KF_STATE_SPACE_VEC std;
    for (uint8_t i = 0; i < KALMAN_STATE_SPACE_DIM; i++) {
        if (i < 4) {
            std(i) = std::max(_init_pos_weight * _std_factor_detection * measurement(i % 2 == 0 ? 2 : 3), _min_std_detection);
        } else {
            std(i) = std::max(_init_vel_weight * _std_factor_detection * measurement(i % 2 == 0 ? 2 : 3), _min_std_detection);
        }
    }
    KF_STATE_SPACE_VEC std_squared = std.array().square();
    KF_STATE_SPACE_MATRIX covariance = std_squared.asDiagonal();
    return std::make_pair(mean_state_space, covariance);
}

void KalmanFilter::predict(KF_STATE_SPACE_VEC &mean, KF_STATE_SPACE_MATRIX &covariance) {
}

}// namespace byte_kalman
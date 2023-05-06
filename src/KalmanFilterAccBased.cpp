#include "KalmanFilterAccBased.h"
#include <eigen3/Eigen/Cholesky>

namespace kalman_modified {
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
    _std_factor_motion_compensated_detection = 0.14;
    _min_std_motion_compensated_detection = 5.0;

    _velocity_coupling_factor = 0.6;
    _velocity_half_life = 2;
}

void KalmanFilter::_init_kf_matrices(double dt) {
    // This is a 4x8 matrix that maps the 8-dimensional state space vector [x, y, w, h, vx, vy, vw, vh]
    // to the 4-dimensional measurement space vector [x, y, w, h]
    _measurement_matrix = Eigen::MatrixXf::Identity(KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM);

    // This is an 8x8 matrix that defines the state transition function.
    // It maps the current state space vector to the next state space vector.
    _state_transition_matrix = Eigen::MatrixXf::Identity(KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM);
    for (uint8_t i = 0; i < 4; i++) {
        _state_transition_matrix(i, i + 4) = _velocity_coupling_factor * dt;
        _state_transition_matrix(i, (i + 2) % 4 + 4) = (1.0 - _velocity_coupling_factor) * dt;
        _state_transition_matrix(i + 4, i + 4) = std::pow(0.5, (dt / _velocity_half_life));
    }

    // This is an 8x8 matrix that defines the process noise covariance matrix.
    // This takes into account acceleration and jerk for modeling the process noise.
    _process_noise_covariance = Eigen::MatrixXf::Identity(KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM);
    for (uint8_t i = 0; i < 4; i++) {
        _process_noise_covariance(i, i) = std::pow(dt, 4) / 4 + std::pow(dt, 2);
        _process_noise_covariance(i, i + 4) = std::pow(dt, 3) / 2;
        _process_noise_covariance(i + 4, i) = std::pow(dt, 3) / 2;
        _process_noise_covariance(i + 4, i + 4) = std::pow(dt, 2);
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
            std(i) = std::max(_init_pos_weight * _std_factor_detection * (i % 2 == 0 ? w : h), _min_std_detection);
        } else {
            std(i) = std::max(_init_vel_weight * _std_factor_detection * (i % 2 == 0 ? w : h), _min_std_detection);
        }
    }
    KFStateSpaceVec std_squared = std.array().square();
    KFStateSpaceMatrix covariance = std_squared.asDiagonal();
    return std::make_pair(mean_state_space, covariance);
}

void KalmanFilter::predict(KFStateSpaceVec &mean, KFStateSpaceMatrix &covariance) {
    float std = _std_factor_acceleration * std::max(mean(2), mean(3)) + _std_offset_acceleration;
    KFMeasSpaceMatrix motion_cov = std::pow(std, 2) * _process_noise_covariance;

    mean = _state_transition_matrix * mean;
    covariance = _state_transition_matrix * covariance * _state_transition_matrix.transpose() + motion_cov;
}

KFDataMeasurementSpace KalmanFilter::project(const KFStateSpaceVec &mean, const KFStateSpaceMatrix &covariance, bool motion_compensated = false) {
    float std_factor = motion_compensated ? _std_factor_motion_compensated_detection : _std_factor_detection;
    float min_std = motion_compensated ? _min_std_motion_compensated_detection : _min_std_detection;

    Eigen::VectorXf measurement_cov(KALMAN_MEASUREMENT_SPACE_DIM);
    measurement_cov << std::max(std_factor * mean(2), min_std),
            std::max(std_factor * mean(3), min_std),
            std::max(std_factor * mean(2), min_std),
            std::max(std_factor * mean(3), min_std);
    measurement_cov = measurement_cov.array().square();
    measurement_cov = measurement_cov.asDiagonal();

    KFMeasSpaceVec mean_updated = _measurement_matrix * mean;
    KFMeasSpaceMatrix covariance_updated = _measurement_matrix * covariance * _measurement_matrix.transpose() + measurement_cov;
    return std::make_pair(mean_updated, covariance_updated);
}

KFDataStateSpace KalmanFilter::update(const KFStateSpaceVec &mean, const KFStateSpaceMatrix &covariance, const DetVec &measurement) {
    KFDataMeasurementSpace projected = project(mean, covariance);
    KFMeasSpaceVec projected_mean = projected.first;
    KFMeasSpaceMatrix projected_covariance = projected.second;

    Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM> B = (covariance * _measurement_matrix.transpose()).transpose();
    Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_MEASUREMENT_SPACE_DIM> kalman_gain = (projected_covariance.llt().solve(B)).transpose();
    Eigen::Matrix<float, 1, KALMAN_MEASUREMENT_SPACE_DIM> innovation = measurement - projected_mean;

    KFStateSpaceVec mean_updated = mean + innovation * kalman_gain.transpose();
    KFStateSpaceMatrix covariance_updated = covariance - kalman_gain * projected_covariance * kalman_gain.transpose();
    return std::make_pair(mean_updated, covariance_updated);
}

}// namespace kalman_modified

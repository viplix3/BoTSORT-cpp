#include "KalmanFilterAccBased.h"

#include <eigen3/Eigen/Cholesky>

namespace acc_kalman
{
KalmanFilter::KalmanFilter(double dt)
    : _init_pos_weight(5.0), _init_vel_weight(15.0),
      _std_factor_acceleration(50.25), _std_offset_acceleration(100.5),
      _std_factor_detection(0.10), _min_std_detection(4.0),
      _std_factor_motion_compensated_detection(0.14),
      _min_std_motion_compensated_detection(5.0),
      _velocity_coupling_factor(0.6), _velocity_half_life(2)
{

    _init_kf_matrices(dt);
}

void KalmanFilter::_init_kf_matrices(double dt)
{
    // This is a 4x8 matrix that maps the 8-dimensional state space vector [x, y, w, h, vx, vy, vw, vh]
    // to the 4-dimensional measurement space vector [x, y, w, h]
    _measurement_matrix.setIdentity();

    // This is a 8x8 matrix that defines the state transition function.
    // It maps the current state space vector to the next state space vector.
    _state_transition_matrix.setIdentity();
    for (Eigen::Index i = 0; i < 4; i++)
    {
        _state_transition_matrix(i, i + 4) =
                static_cast<float>(_velocity_coupling_factor * dt);
        _state_transition_matrix(i, (i + 2) % 4 + 4) =
                static_cast<float>((1.0F - _velocity_coupling_factor) * dt);
        _state_transition_matrix(i + 4, i + 4) =
                static_cast<float>(std::pow(0.5, (dt / _velocity_half_life)));
    }

    // This is a 8x8 matrix that defines the process noise covariance matrix.
    // This takes into account acceleration and jerk for modeling the process noise.
    _process_noise_covariance = Eigen::MatrixXf::Identity(
            KALMAN_STATE_SPACE_DIM, KALMAN_STATE_SPACE_DIM);
    for (Eigen::Index i = 0; i < 4; i++)
    {
        _process_noise_covariance(i, i) =
                static_cast<float>(std::pow(dt, 4) / 4 + std::pow(dt, 2));
        _process_noise_covariance(i, i + 4) =
                static_cast<float>(std::pow(dt, 3) / 2);
        _process_noise_covariance(i + 4, i) =
                static_cast<float>(std::pow(dt, 3) / 2);
        _process_noise_covariance(i + 4, i + 4) =
                static_cast<float>(std::pow(dt, 2));
    }
}

KFDataStateSpace KalmanFilter::init(const DetVec &measurement) const
{
    constexpr float init_velocity = 0.0;

    KFStateSpaceVec mean_state_space;
    mean_state_space.head<4>() = measurement.head<4>();
    mean_state_space.tail<4>().setConstant(init_velocity);

    float w = measurement(2), h = measurement(3);
    KFStateSpaceVec std_dev;
    std_dev.head<4>() = (_init_pos_weight * _std_factor_detection *
                         Eigen::Vector4f(w, h, w, h))
                                .cwiseMax(_min_std_detection);
    std_dev.tail<4>() = (_init_vel_weight * _std_factor_detection *
                         Eigen::Vector4f(w, h, w, h))
                                .cwiseMax(_min_std_detection);

    KFStateSpaceMatrix covariance =
            std_dev.array().square().matrix().asDiagonal();
    return std::make_pair(mean_state_space, covariance);
}

void KalmanFilter::predict(KFStateSpaceVec &mean,
                           KFStateSpaceMatrix &covariance)
{
    float std = _std_factor_acceleration * std::max(mean(2), mean(3)) +
                _std_offset_acceleration;
    KFStateSpaceMatrix motion_cov =
            std::pow(std, 2) * _process_noise_covariance;

    mean = _state_transition_matrix * mean.transpose();
    covariance = _state_transition_matrix * covariance *
                         _state_transition_matrix.transpose() +
                 motion_cov;
}

KFDataMeasurementSpace
KalmanFilter::project(const KFStateSpaceVec &mean,
                      const KFStateSpaceMatrix &covariance,
                      bool motion_compensated) const
{
    float std_factor = motion_compensated
                               ? _std_factor_motion_compensated_detection
                               : _std_factor_detection;
    float min_std = motion_compensated ? _min_std_motion_compensated_detection
                                       : _min_std_detection;

    Eigen::Vector4f std;
    std << std::max(std_factor * mean(2), min_std),
            std::max(std_factor * mean(3), min_std),
            std::max(std_factor * mean(2), min_std),
            std::max(std_factor * mean(3), min_std);
    KFMeasSpaceMatrix measurement_cov = KFMeasSpaceMatrix::Zero();
    measurement_cov.diagonal() = std.array().square();

    KFMeasSpaceVec mean_projected = _measurement_matrix * mean.transpose();
    KFMeasSpaceMatrix covariance_projected =
            _measurement_matrix * covariance * _measurement_matrix.transpose() +
            measurement_cov;
    return std::make_pair(mean_projected, covariance_projected);
}

KFDataStateSpace KalmanFilter::update(const KFStateSpaceVec &mean,
                                      const KFStateSpaceMatrix &covariance,
                                      const DetVec &measurement)
{
    KFDataMeasurementSpace projected = project(mean, covariance);
    KFMeasSpaceVec projected_mean = projected.first;
    KFMeasSpaceMatrix projected_covariance = projected.second;

    Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM>
            B = (covariance * _measurement_matrix.transpose()).transpose();
    Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_MEASUREMENT_SPACE_DIM>
            kalman_gain = (projected_covariance.llt().solve(B)).transpose();
    Eigen::Matrix<float, 1, KALMAN_MEASUREMENT_SPACE_DIM> innovation =
            measurement - projected_mean;

    KFStateSpaceVec mean_updated = mean + innovation * kalman_gain.transpose();
    KFStateSpaceMatrix covariance_updated =
            covariance -
            kalman_gain * projected_covariance * kalman_gain.transpose();
    return std::make_pair(mean_updated, covariance_updated);
}


Eigen::Matrix<float, 1, Eigen::Dynamic> KalmanFilter::gating_distance(
        const KFStateSpaceVec &mean, const KFStateSpaceMatrix &covariance,
        const std::vector<DetVec> &measurements, bool only_position) const
{
    KFDataMeasurementSpace projected = this->project(mean, covariance);
    KFMeasSpaceVec projected_mean = projected.first;
    KFMeasSpaceMatrix projected_covariance = projected.second;

    if (only_position)
    {
        projected_mean.tail<4>().setZero();
        projected_covariance.bottomRightCorner<2, 2>().setZero();
    }

    Eigen::LLT<Eigen::MatrixXf> lltOfProjectedCovariance(projected_covariance);
    Eigen::Matrix<float, 1, Eigen::Dynamic> mahalanobis_distances(
            measurements.size());
    mahalanobis_distances.setZero();

    for (Eigen::Index i = 0; i < measurements.size(); i++)
    {
        Eigen::VectorXf diff = measurements[i] - projected_mean;
        // Solve for y in Ly = diff using forward substitution, more efficient than computing the inverse
        Eigen::VectorXf y = lltOfProjectedCovariance.matrixL().solve(diff);
        // Mahalanobis distance is the norm of y
        mahalanobis_distances(i) = y.squaredNorm();
    }

    return mahalanobis_distances;
}
}// namespace acc_kalman

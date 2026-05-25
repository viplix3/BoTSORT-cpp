#include "PitchKalmanFilter.h"

#include <cmath>
#include <eigen3/Eigen/Cholesky>

namespace bot_kalman {

PitchKalmanFilter::PitchKalmanFilter(float dt,
                                     float std_weight_pos,
                                     float std_weight_vel)
    : _dt(std::max(dt, 1e-3F))
    , _std_weight_position_m(std_weight_pos)
    , _std_weight_velocity_m(std_weight_vel)
{
    _init_matrices(_dt);
}

void PitchKalmanFilter::_init_matrices(float dt) {
    _measurement_matrix.setZero();
    _measurement_matrix(0, 0) = 1.0F;  // measure x
    _measurement_matrix(1, 1) = 1.0F;  // measure y

    _state_transition_matrix.setIdentity();
    _state_transition_matrix(0, 2) = dt;  // x += vx * dt
    _state_transition_matrix(1, 3) = dt;  // y += vy * dt
}

PKFDataState PitchKalmanFilter::init(const PKFMeasVec& measurement) const {
    PKFStateVec mean = PKFStateVec::Zero();
    mean(0) = measurement(0);
    mean(1) = measurement(1);

    PKFStateVec std_dev;
    std_dev(0) = 2.0F * _std_weight_position_m;
    std_dev(1) = 2.0F * _std_weight_position_m;
    std_dev(2) = 10.0F * _std_weight_velocity_m;
    std_dev(3) = 10.0F * _std_weight_velocity_m;

    PKFStateMat covariance = std_dev.array().square().matrix().asDiagonal();
    return {mean, covariance};
}

void PitchKalmanFilter::predict(PKFStateVec& mean,
                                PKFStateMat& covariance) const {
    PKFStateVec std_dev;
    std_dev(0) = _std_weight_position_m;
    std_dev(1) = _std_weight_position_m;
    std_dev(2) = _std_weight_velocity_m;
    std_dev(3) = _std_weight_velocity_m;
    PKFStateMat motion_cov = std_dev.array().square().matrix().asDiagonal();

    mean = _state_transition_matrix * mean;
    covariance = _state_transition_matrix * covariance *
                 _state_transition_matrix.transpose() + motion_cov;
}

PKFDataMeas PitchKalmanFilter::project(const PKFStateVec& mean,
                                       const PKFStateMat& covariance) const {
    PKFMeasVec innovation_std;
    innovation_std << _std_weight_position_m, _std_weight_position_m;
    PKFMeasMat innovation_cov = innovation_std.array().square().matrix().asDiagonal();

    PKFMeasVec projected_mean = _measurement_matrix * mean;
    PKFMeasMat projected_cov  = _measurement_matrix * covariance *
                                _measurement_matrix.transpose() + innovation_cov;
    return {projected_mean, projected_cov};
}

PKFDataState PitchKalmanFilter::update(const PKFStateVec& mean,
                                       const PKFStateMat& covariance,
                                       const PKFMeasVec& measurement) const {
    // Non-finite-measurement guard (design §4 edge cases).
    if (!measurement.allFinite()) {
        return {mean, covariance};
    }
    auto [proj_mean, proj_cov] = project(mean, covariance);

    Eigen::Matrix<float, PITCH_KF_MEAS_DIM, PITCH_KF_STATE_DIM> B =
        (covariance * _measurement_matrix.transpose()).transpose();
    Eigen::Matrix<float, PITCH_KF_STATE_DIM, PITCH_KF_MEAS_DIM> gain =
        proj_cov.llt().solve(B).transpose();
    PKFMeasVec innovation = measurement - proj_mean;

    PKFStateVec updated_mean = mean + gain * innovation;
    PKFStateMat updated_cov  = covariance - gain * proj_cov * gain.transpose();
    return {updated_mean, updated_cov};
}

Eigen::Matrix<float, 1, Eigen::Dynamic> PitchKalmanFilter::gating_distance(
    const PKFStateVec& mean,
    const PKFStateMat& covariance,
    const std::vector<PKFMeasVec>& measurements) const {

    auto [proj_mean, proj_cov] = project(mean, covariance);
    Eigen::LLT<Eigen::MatrixXf> llt(proj_cov);
    Eigen::Matrix<float, 1, Eigen::Dynamic> dists(measurements.size());
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(measurements.size()); ++i) {
        Eigen::VectorXf diff = measurements[i] - proj_mean;
        Eigen::VectorXf y = llt.matrixL().solve(diff);
        dists(i) = y.squaredNorm();
    }
    return dists;
}

}  // namespace bot_kalman

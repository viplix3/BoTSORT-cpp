#pragma once

/// @file PitchKalmanFilter.h
/// @brief 4-D metre-space Kalman filter for camera-pan-invariant tracking.
///
/// Phase E.2 of the OpticXI tracking pipeline. Runs in parallel with the
/// existing 8-D pixel KalmanFilter; the host bank decides per-frame which
/// gate to use based on homography validity. State is foot-point only:
/// width / height stay in pixels (perspective foreshortening makes
/// width-in-metres physically poorly defined). Constant-velocity model;
/// process / measurement noise scaled by metre-space standard deviations.

#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>

namespace bot_kalman {

inline constexpr int PITCH_KF_STATE_DIM = 4;
inline constexpr int PITCH_KF_MEAS_DIM  = 2;

using PKFStateVec = Eigen::Matrix<float, PITCH_KF_STATE_DIM, 1>;
using PKFStateMat = Eigen::Matrix<float, PITCH_KF_STATE_DIM, PITCH_KF_STATE_DIM>;
using PKFMeasVec  = Eigen::Matrix<float, PITCH_KF_MEAS_DIM, 1>;
using PKFMeasMat  = Eigen::Matrix<float, PITCH_KF_MEAS_DIM, PITCH_KF_MEAS_DIM>;

using PKFDataState = std::pair<PKFStateVec, PKFStateMat>;
using PKFDataMeas  = std::pair<PKFMeasVec,  PKFMeasMat>;

class PitchKalmanFilter {
public:
    /// Chi-squared 95th percentile, 2 DoF (foot-point measurement). Used
    /// as the Mahalanobis gating threshold.
    static constexpr float chi2inv95 = 5.9915F;

    /// @param dt              Frame interval in seconds (1 / fps).
    /// @param std_weight_pos  Position standard deviation, metres.
    /// @param std_weight_vel  Velocity standard deviation, m/s.
    PitchKalmanFilter(float dt, float std_weight_pos, float std_weight_vel);

    /// Initialize from a measurement; velocity starts at zero.
    PKFDataState init(const PKFMeasVec& measurement) const;

    /// Constant-velocity predict step (in place).
    void predict(PKFStateVec& mean, PKFStateMat& covariance) const;

    /// Project (mean, covariance) to measurement space.
    PKFDataMeas project(const PKFStateVec& mean,
                        const PKFStateMat& covariance) const;

    /// Standard Kalman update step.
    /// If `measurement` contains any non-finite value, returns
    /// (mean, covariance) unchanged.
    PKFDataState update(const PKFStateVec& mean,
                        const PKFStateMat& covariance,
                        const PKFMeasVec& measurement) const;

    /// Squared Mahalanobis distance for each candidate measurement.
    Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance(
        const PKFStateVec& mean,
        const PKFStateMat& covariance,
        const std::vector<PKFMeasVec>& measurements) const;

private:
    void _init_matrices(float dt);

    float _dt;
    float _std_weight_position_m;
    float _std_weight_velocity_m;
    PKFStateMat _state_transition_matrix;
    Eigen::Matrix<float, PITCH_KF_MEAS_DIM, PITCH_KF_STATE_DIM> _measurement_matrix;
};

}  // namespace bot_kalman

#include "matching.h"

/**
 * @brief Calculate the cosine distance between two feature vectors
 * 
 * @param x Feature vector 1
 * @param y Feature vector 2
 * @return float Cosine distance (1 - cosine similarity)
 */
inline float cosine_distance(const FeatureVector &x, const FeatureVector &y) {
    return 1.0f - (x.dot(y) / (x.norm() * y.norm() + 1e-5f));
}

CostMatrix embedding_distance(const std::vector<Track *> &tracks, const std::vector<Track *> &detections) {
    int num_tracks = tracks.size();
    int num_detections = detections.size();

    CostMatrix cost_matrix = Eigen::MatrixXf::Zero(num_tracks, num_detections);
    if (num_tracks > 0 && num_detections > 0) {
        for (int i = 0; i < num_tracks; i++) {
            for (int j = 0; j < num_detections; j++) {
                cost_matrix(i, j) = cosine_distance(tracks[i]->smooth_feat, detections[j]->curr_feat);
            }
        }
    }

    return cost_matrix;
}


CostMatrix fuse_motion(KalmanFilter &KF,
                       CostMatrix &cost_matrix,
                       std::vector<Track *> tracks,
                       std::vector<Track *> detections,
                       bool only_position) {
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        return cost_matrix;
    }

    uint8_t gating_dim = (only_position == true) ? 2 : 4;
    const double gating_threshold = KalmanFilter::chi2inv95[gating_dim];

    for (size_t i = 0; i < tracks.size(); i++) {
        KF.gating_distance(tracks[i]->mean, tracks[i]->covariance, detections, only_position);
    }
}
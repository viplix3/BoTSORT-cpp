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

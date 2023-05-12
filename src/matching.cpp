#include "matching.h"
#include "utils.h"

CostMatrix iou_distance(const std::vector<Track *> &tracks, const std::vector<Track *> &detections) {
    int num_tracks = tracks.size();
    int num_detections = detections.size();

    CostMatrix cost_matrix = Eigen::MatrixXf::Zero(num_tracks, num_detections);
    if (num_tracks > 0 && num_detections > 0) {
        for (int i = 0; i < num_tracks; i++) {
            for (int j = 0; j < num_detections; j++) {
                cost_matrix(i, j) = 1.0 - iou(tracks[i]->get_tlwh(), detections[j]->get_tlwh());
            }
        }
    }

    return cost_matrix;
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

void fuse_motion(KalmanFilter &KF,
                 CostMatrix &cost_matrix,
                 std::vector<Track *> tracks,
                 std::vector<Track *> detections,
                 bool only_position,
                 float lambda) {
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        return;
    }

    uint8_t gating_dim = (only_position == true) ? 2 : 4;
    const double gating_threshold = KalmanFilter::chi2inv95[gating_dim];

    std::vector<DetVec> measurements;
    std::vector<float> det_xywh;
    for (size_t i = 0; i < detections.size(); i++) {
        DetVec det;

        det_xywh = detections[i]->get_tlwh();
        det << det_xywh[0], det_xywh[1], det_xywh[2], det_xywh[3];
        measurements.emplace_back(det);
    }

    for (size_t i = 0; i < tracks.size(); i++) {
        Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance = KF.gating_distance(
                tracks[i]->mean,
                tracks[i]->covariance,
                measurements,
                only_position);

        for (size_t j = 0; j < gating_distance.size(); j++) {
            if (gating_distance(0, j) > gating_threshold) {
                cost_matrix(i, j) = std::numeric_limits<float>::infinity();
            }

            cost_matrix(i, j) = lambda * cost_matrix(i, j) + (1 - lambda) * gating_distance[j];
        }
    }
}

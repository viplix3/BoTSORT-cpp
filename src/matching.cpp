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
                cost_matrix(i, j) = std::max(0.0f, cosine_distance(tracks[i]->smooth_feat, detections[j]->curr_feat));
            }
        }
    }

    return cost_matrix;
}

void fuse_score(CostMatrix &cost_matrix, std::vector<Track *> detections) {
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0) {
        return;
    }

    for (size_t i = 0; i < cost_matrix.rows(); i++) {
        for (size_t j = 0; j < cost_matrix.cols(); j++) {
            cost_matrix(i, j) = 1.0 - ((1.0 - cost_matrix(i, j)) * detections[j]->get_score());
        }
    }
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

CostMatrix fuse_iou_with_emb(CostMatrix &iou_matrix, CostMatrix &emb_matrix, float iou_threshold, float appearance_threshold) {
    if (emb_matrix.rows() == 0 || emb_matrix.cols() == 0) {
        return iou_matrix;
    }

    // If iou distance is larger than threshold, set emb distance to inf
    for (size_t i = 0; i < iou_matrix.rows(); i++) {
        for (size_t j = 0; j < iou_matrix.cols(); j++) {
            if (iou_matrix(i, j) < iou_threshold) {
                emb_matrix(i, j) = std::numeric_limits<float>::infinity();
            }
        }
    }

    // If emb distance is larger than threshold, set the emb distance to inf
    for (size_t i = 0; i < emb_matrix.rows(); i++) {
        for (size_t j = 0; j < emb_matrix.cols(); j++) {
            if (emb_matrix(i, j) > appearance_threshold) {
                emb_matrix(i, j) = std::numeric_limits<float>::infinity();
            }
        }
    }

    // Fuse iou and emb distance by taking the element-wise minimum
    CostMatrix cost_matrix = Eigen::MatrixXf::Zero(iou_matrix.rows(), iou_matrix.cols());
    for (size_t i = 0; i < iou_matrix.rows(); i++) {
        for (size_t j = 0; j < iou_matrix.cols(); j++) {
            cost_matrix(i, j) = std::min(iou_matrix(i, j), emb_matrix(i, j));
        }
    }

    return cost_matrix;
}
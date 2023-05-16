#pragma once

#include "DataType.h"
#include "track.h"

/**
 * @brief Calculate the IoU distance (1 - IoU) between tracks and detections
 * 
 * @param tracks Tracks
 * @param detections Tracks created from detections
 * @return CostMatrix IoU distance matrix
 */
CostMatrix iou_distance(const std::vector<std::shared_ptr<Track>> &tracks, const std::vector<std::shared_ptr<Track>> &detections);

/**
 * @brief Calculate the embedding distance between tracks and detections using cosine distance
 * 
 * @param tracks Confirmed tracks
 * @param detections Tracks created from detections
 * @return CostMatrix Embedding distance matrix
 */
CostMatrix embedding_distance(const std::vector<std::shared_ptr<Track>> &tracks, const std::vector<std::shared_ptr<Track>> &detections);

/**
 * @brief Fuses the detection score into the cost matrix in-place
 *     fused_cost = 1 - ((1 - cost_matrix) * detection_score)
 *     fused_cost = 1 - (similarity * detection_score)
 * 
 * @param cost_matrix Cost matrix in which to fuse the detection score
 * @param detections Tracks created from detections used to create the cost matrix
 */
void fuse_score(CostMatrix &cost_matrix, const std::vector<std::shared_ptr<Track>> &detections);

/**
 * @brief Fuses motion (maha distance) into the cost matrix in-place
 *      fused_cost = lambda * cost_matrix + (1 - lambda) * motion_distance
 * @param KF Kalman filter
 * @param cost_matrix Cost matrix in which to fuse motion
 * @param tracks Tracks used to create the cost matrix
 * @param detections Tracks created from detections used to create the cost matrix
 * @param only_position Set to true only position should be used for gating distance
 * @param lambda Weighting factor for motion
 */
void fuse_motion(const KalmanFilter &KF,
                 CostMatrix &cost_matrix,
                 const std::vector<std::shared_ptr<Track>> &tracks,
                 const std::vector<std::shared_ptr<Track>> &detections,
                 bool only_position = false,
                 float lambda = 0.98F);

/**
 * @brief Fuses IoU and embedding distance into a single cost matrix
 * 
 * @param iou_matrix IoU distance matrix
 * @param emb_matrix Embedding distance matrix
 * @param iou_threshold Threshold for IoU distance
 * @param appearance_threshold Threshold for embedding distance
 * @return CostMatrix Fused cost matrix
 */
CostMatrix fuse_iou_with_emb(CostMatrix &iou_matrix, CostMatrix &emb_matrix, float iou_threshold, float appearance_threshold);

void linear_assignment(CostMatrix &cost_matrix, float thresh, AssociationData &associations);
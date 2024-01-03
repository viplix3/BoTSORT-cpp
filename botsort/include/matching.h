#pragma once

#include <iostream>
#include <tuple>

#include "DataType.h"
#include "track.h"

/**
 * @brief Calculate the IoU distance between tracks and detections and create a mask for the cost matrix
 *  when the IoU distance is greater than the threshold
 * 
 * @param tracks Tracks used to create the cost matrix
 * @param detections Tracks created from detections used to create the cost matrix
 * @param max_iou_distance Threshold for IoU distance
 * @return std::tuple<CostMatrix, CostMatrix> Tuple of IoU distance cost matrix and IoU distance mask
 */
std::tuple<CostMatrix, CostMatrix>
iou_distance(const std::vector<std::shared_ptr<Track>> &tracks,
             const std::vector<std::shared_ptr<Track>> &detections,
             float max_iou_distance);

/**
 * @brief Calculate the IoU distance between tracks and detections
 * 
 * @param tracks Tracks used to create the cost matrix
 * @param detections Tracks created from detections used to create the cost matrix
 * @return CostMatrix IoU distance cost matrix
 */
CostMatrix iou_distance(const std::vector<std::shared_ptr<Track>> &tracks,
                        const std::vector<std::shared_ptr<Track>> &detections);


/**
 * @brief Calculate the embedding distance between tracks and detections and create a mask for the cost matrix
 *  when the embedding distance is greater than the threshold
 * 
 * @param tracks Tracks used to create the cost matrix
 * @param detections Tracks created from detections used to create the cost matrix
 * @param max_embedding_distance Threshold for embedding distance
 * @param distance_metric Distance metric to use for calculating the embedding distance
 * @return std::tuple<CostMatrix, CostMatrix> Tuple of embedding distance cost matrix and embedding distance mask
 */
std::tuple<CostMatrix, CostMatrix>
embedding_distance(const std::vector<std::shared_ptr<Track>> &tracks,
                   const std::vector<std::shared_ptr<Track>> &detections,
                   float max_embedding_distance,
                   const std::string &distance_metric);

/**
 * @brief Fuses the detection score into the cost matrix in-place
 *     fused_cost = 1 - ((1 - cost_matrix) * detection_score)
 *     fused_cost = 1 - (similarity * detection_score)
 * 
 * @param cost_matrix Cost matrix in which to fuse the detection score
 * @param detections Tracks created from detections used to create the cost matrix
 */
void fuse_score(CostMatrix &cost_matrix,
                const std::vector<std::shared_ptr<Track>> &detections);

/**
 * @brief Fuses motion (maha distance) into the cost matrix in-place
 *      fused_cost = lambda * cost_matrix + (1 - lambda) * motion_distance
 * @param KF Kalman filter
 * @param cost_matrix Cost matrix in which to fuse motion
 * @param tracks Tracks used to create the cost matrix
 * @param detections Tracks created from detections used to create the cost matrix
 * @param lambda Weighting factor for motion (default: 0.98)
 * @param only_position Set to true only position should be used for gating distance
 */
void fuse_motion(const KalmanFilter &KF, CostMatrix &cost_matrix,
                 const std::vector<std::shared_ptr<Track>> &tracks,
                 const std::vector<std::shared_ptr<Track>> &detections,
                 float lambda = 0.98F, bool only_position = false);

/**
 * @brief Fuse IoU distance with embedding distance keeping the mask in mind
 * 
 * @param iou_dist Score fused IoU distance cost matrix
 * @param emb_dist Motion fused embedding distance cost matrix
 * @param iou_dists_mask IoU distance mask
 * @param emb_dists_mask Embedding distance mask
 * @return CostMatrix Fused and masked cost matrix
 */
CostMatrix fuse_iou_with_emb(CostMatrix &iou_dist, CostMatrix &emb_dist,
                             const CostMatrix &iou_dists_mask,
                             const CostMatrix &emb_dists_mask);

/**
 * @brief Performs linear assignment using the LAPJV algorithm
 * 
 * @param cost_matrix Cost matrix for solving the linear assignment problem
 * @param thresh Threshold for cost matrix
 * @return AssociationData Association data
 */
AssociationData linear_assignment(CostMatrix &cost_matrix, float thresh);
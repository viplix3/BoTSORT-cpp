#pragma once

#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

#include "DataType.h"
#include "PitchKalmanFilter.h"
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
 * @brief Phase E.2 (OpticXI) — additive metre-space Mahalanobis gate.
 *
 * For each (track i, detection j) where TRACK has pitch_kf_initialized()
 * AND DETECTION has a metre measurement (the optional at
 * metre_measurements[j] is engaged), compute the metre Mahalanobis
 * distance via @p pitch_kf and override cost_matrix(i, j) to +infinity
 * if it exceeds PitchKalmanFilter::chi2inv95. Cells where either side
 * lacks metre state are LEFT UNCHANGED, so the upstream pixel gate's
 * decision stands.
 *
 * No-ops (returns early) when the input matrices are empty or when the
 * metre_measurements vector size doesn't match detections.size() (caller
 * contract violation — fail-safe to leave the cost matrix untouched).
 *
 * @param pitch_kf            The metre-space KF (caller owns).
 * @param cost_matrix         In/out. Same shape as tracks × detections.
 * @param tracks              The tracks rows correspond to.
 * @param detections          The detections cols correspond to (only the
 *                            size is checked here; the measurement values
 *                            come from metre_measurements).
 * @param metre_measurements  Parallel to detections (size must equal
 *                            detections.size()). std::nullopt entries
 *                            are skipped (cell left unchanged).
 */
void fuse_motion_pitch(
    const PitchKalmanFilter &pitch_kf,
    CostMatrix &cost_matrix,
    const std::vector<std::shared_ptr<Track>> &tracks,
    const std::vector<std::shared_ptr<Track>> &detections,
    const std::vector<std::optional<bot_kalman::PKFMeasVec>> &metre_measurements);

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
#include "DataType.h"
#include "track.h"

/**
 * @brief Calculate the embedding distance between tracks and detections using cosine distance
 * 
 * @param tracks Confirmed tracks
 * @param detections Tracks created from detections
 * @return CostMatrix Embedding distance matrix
 */
CostMatrix embedding_distance(const std::vector<Track *> &tracks, const std::vector<Track *> &detections);

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
void fuse_motion(KalmanFilter &KF,
                 CostMatrix &cost_matrix,
                 std::vector<Track *> tracks,
                 std::vector<Track *> detections,
                 bool only_position = false,
                 float lambda = 0.98);
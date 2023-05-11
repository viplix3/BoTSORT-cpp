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

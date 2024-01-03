#include "matching.h"

#include "DataType.h"
#include "utils.h"

std::tuple<CostMatrix, CostMatrix>
iou_distance(const std::vector<std::shared_ptr<Track>> &tracks,
             const std::vector<std::shared_ptr<Track>> &detections,
             float max_iou_distance)
{
    size_t num_tracks = tracks.size();
    size_t num_detections = detections.size();

    CostMatrix cost_matrix =
            Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_tracks),
                                  static_cast<Eigen::Index>(num_detections));
    CostMatrix iou_dists_mask =
            Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_tracks),
                                  static_cast<Eigen::Index>(num_detections));

    if (num_tracks > 0 && num_detections > 0)
    {
        for (int i = 0; i < num_tracks; i++)
        {
            for (int j = 0; j < num_detections; j++)
            {
                cost_matrix(i, j) = 1.0F - iou(tracks[i]->get_tlwh(),
                                               detections[j]->get_tlwh());

                if (cost_matrix(i, j) > max_iou_distance)
                {
                    iou_dists_mask(i, j) = 1.0F;
                }
            }
        }
    }

    return {cost_matrix, iou_dists_mask};
}

CostMatrix iou_distance(const std::vector<std::shared_ptr<Track>> &tracks,
                        const std::vector<std::shared_ptr<Track>> &detections)
{
    size_t num_tracks = tracks.size();
    size_t num_detections = detections.size();

    CostMatrix cost_matrix =
            Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_tracks),
                                  static_cast<Eigen::Index>(num_detections));
    if (num_tracks > 0 && num_detections > 0)
    {
        for (int i = 0; i < num_tracks; i++)
        {
            for (int j = 0; j < num_detections; j++)
            {
                cost_matrix(i, j) = 1.0F - iou(tracks[i]->get_tlwh(),
                                               detections[j]->get_tlwh());
            }
        }
    }

    return cost_matrix;
}

std::tuple<CostMatrix, CostMatrix>
embedding_distance(const std::vector<std::shared_ptr<Track>> &tracks,
                   const std::vector<std::shared_ptr<Track>> &detections,
                   float max_embedding_distance,
                   const std::string &distance_metric)
{
    if (!(distance_metric == "euclidean" || distance_metric == "cosine"))
    {
        std::cout << "Invalid distance metric " << distance_metric
                  << " passed.";
        std::cout << "Only 'euclidean' and 'cosine' are supported."
                  << std::endl;
        exit(1);
    }

    size_t num_tracks = tracks.size();
    size_t num_detections = detections.size();

    CostMatrix cost_matrix =
            Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_tracks),
                                  static_cast<Eigen::Index>(num_detections));
    CostMatrix embedding_dists_mask =
            Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(num_tracks),
                                  static_cast<Eigen::Index>(num_detections));

    if (num_tracks > 0 && num_detections > 0)
    {
        for (int i = 0; i < num_tracks; i++)
        {
            for (int j = 0; j < num_detections; j++)
            {
                if (distance_metric == "euclidean")
                    cost_matrix(i, j) = std::max(
                            0.0f, euclidean_distance(tracks[i]->smooth_feat,
                                                     detections[j]->curr_feat));
                else
                    cost_matrix(i, j) = std::max(
                            0.0f, cosine_distance(tracks[i]->smooth_feat,
                                                  detections[j]->curr_feat));

                if (cost_matrix(i, j) > max_embedding_distance)
                {
                    embedding_dists_mask(i, j) = 1.0F;
                }
            }
        }
    }

    return {cost_matrix, embedding_dists_mask};
}

void fuse_score(CostMatrix &cost_matrix,
                const std::vector<std::shared_ptr<Track>> &detections)
{
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0)
    {
        return;
    }

    for (Eigen::Index i = 0; i < cost_matrix.rows(); i++)
    {
        for (Eigen::Index j = 0; j < cost_matrix.cols(); j++)
        {
            cost_matrix(i, j) = 1.0F - ((1.0F - cost_matrix(i, j)) *
                                        detections[j]->get_score());
        }
    }
}

void fuse_motion(const KalmanFilter &KF, CostMatrix &cost_matrix,
                 const std::vector<std::shared_ptr<Track>> &tracks,
                 const std::vector<std::shared_ptr<Track>> &detections,
                 float lambda, bool only_position)
{
    if (cost_matrix.rows() == 0 || cost_matrix.cols() == 0)
    {
        return;
    }

    uint8_t gating_dim = only_position ? 2 : 4;
    const double gating_threshold = KalmanFilter::chi2inv95[gating_dim];

    std::vector<DetVec> measurements;
    std::vector<float> det_xywh;
    for (const std::shared_ptr<Track> &detection: detections)
    {
        DetVec det;

        det_xywh = detection->get_tlwh();
        det << det_xywh[0], det_xywh[1], det_xywh[2], det_xywh[3];
        measurements.emplace_back(det);
    }

    for (Eigen::Index i = 0; i < tracks.size(); i++)
    {
        Eigen::Matrix<float, 1, Eigen::Dynamic> gating_distance =
                KF.gating_distance(tracks[i]->mean, tracks[i]->covariance,
                                   measurements, only_position);

        for (Eigen::Index j = 0; j < gating_distance.size(); j++)
        {
            if (gating_distance(0, j) > gating_threshold)
            {
                cost_matrix(i, j) = std::numeric_limits<float>::infinity();
            }

            cost_matrix(i, j) = lambda * cost_matrix(i, j) +
                                (1 - lambda) * gating_distance[j];
        }
    }
}

CostMatrix fuse_iou_with_emb(CostMatrix &iou_dist, CostMatrix &emb_dist,
                             const CostMatrix &iou_dists_mask,
                             const CostMatrix &emb_dists_mask)
{

    if (emb_dist.rows() == 0 || emb_dist.cols() == 0)
    {
        // Embedding distance is not available, mask off iou distance
        for (Eigen::Index i = 0; i < iou_dist.rows(); i++)
        {
            for (Eigen::Index j = 0; j < iou_dist.cols(); j++)
            {
                if (static_cast<bool>(iou_dists_mask(i, j)))
                {
                    iou_dist(i, j) = 1.0F;
                }
            }
        }
        return iou_dist;
    }

    // If IoU distance is larger than threshold, don't use embedding at all
    for (Eigen::Index i = 0; i < iou_dist.rows(); i++)
    {
        for (Eigen::Index j = 0; j < iou_dist.cols(); j++)
        {
            if (static_cast<bool>(iou_dists_mask(i, j)))
            {
                emb_dist(i, j) = 1.0F;
            }
        }
    }

    // If emb distance is larger than threshold, set the emb distance to inf
    for (Eigen::Index i = 0; i < emb_dist.rows(); i++)
    {
        for (Eigen::Index j = 0; j < emb_dist.cols(); j++)
        {
            if (static_cast<bool>(emb_dists_mask(i, j)))
            {
                emb_dist(i, j) = 1.0F;
            }
        }
    }

    // Fuse iou and emb distance by taking the element-wise minimum
    CostMatrix cost_matrix =
            Eigen::MatrixXf::Zero(iou_dist.rows(), iou_dist.cols());
    for (Eigen::Index i = 0; i < iou_dist.rows(); i++)
    {
        for (Eigen::Index j = 0; j < iou_dist.cols(); j++)
        {
            cost_matrix(i, j) = std::min(iou_dist(i, j), emb_dist(i, j));
        }
    }

    return cost_matrix;
}

AssociationData linear_assignment(CostMatrix &cost_matrix, float thresh)
{
    // If cost matrix is empty, all the tracks and detections are unmatched
    AssociationData associations;

    if (cost_matrix.size() == 0)
    {
        for (int i = 0; i < cost_matrix.rows(); i++)
        {
            associations.unmatched_track_indices.emplace_back(i);
        }

        for (int i = 0; i < cost_matrix.cols(); i++)
        {
            associations.unmatched_det_indices.emplace_back(i);
        }

        return associations;
    }

    std::vector<int> rowsol, colsol;
    double total_cost = lapjv(cost_matrix, rowsol, colsol, true, thresh);

    for (int i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            associations.matches.emplace_back(i, rowsol[i]);
        }
        else
        {
            associations.unmatched_track_indices.emplace_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            associations.unmatched_det_indices.emplace_back(i);
        }
    }

    return associations;
}
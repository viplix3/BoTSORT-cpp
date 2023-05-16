#pragma once

#include "GlobalMotionCompensation.h"
#include "ReID.h"
#include "track.h"


#include <string>


class BoTSORT {
public:
    /**
     * @brief Track the objects in the frame
     * 
     * @param detections Detections in the frame
     * @param frame Frame
     * @return std::vector<std::shared_ptr<Track>> 
     */
    std::vector<std::shared_ptr<Track>> track(const std::vector<Detection> &detections, const cv::Mat &frame);

private:
    bool _reid_enabled;
    uint8_t _track_buffer, _frame_rate, _buffer_size, _max_time_lost;
    float _track_high_thresh, _new_track_thresh, _match_thresh, _proximity_thresh, _appearance_thresh, _lambda;
    unsigned int _frame_id;

    std::vector<std::shared_ptr<Track>> _tracked_tracks;
    std::vector<std::shared_ptr<Track>> _lost_tracks;
    std::vector<std::shared_ptr<Track>> _removed_tracks;

    std::unique_ptr<KalmanFilter> _kalman_filter;
    std::unique_ptr<GlobalMotionCompensation> _gmc_algo;
    std::unique_ptr<ReIDModel> _reid_model;


public:
    /**
     * @brief Construct a new BoTSORT MultiObjectTracking algorithm object
     * 
     * @param model_weights (Optional) Path to the model weights file. If not provided, Re-ID is disabled (default: std::nullopt)
     * @param fp16_inference If true, use FP16 inference (default: false)
     * @param track_high_thresh Detection confidence threshold for classifying a detection as a high-confidence detection (default: 0.7)
     * @param new_track_thresh Detection confidence threshold for creating a new track (default: 0.8)
     * @param track_buffer 
     * @param match_thresh IoU + Re-ID matching threshold for first stage matching (default: 0.7)
     * @param proximity_thresh Minimum IoU threshold for using visual features for matching (default: 0.5)
     * @param appearance_thresh Appearance matching threshold (default: 0.25)
     * @param gmc_method Global motion compensation method (default: "sparseOptFlow")
     * @param frame_rate Frame rate of the video (default: 30)
     * @param lambda Used for fusing motion distance and appearance distance (default: 0.985)
     */
    explicit BoTSORT(
            std::optional<const char *> model_weights = std::nullopt,
            bool fp16_inference = false,
            float track_high_thresh = 0.6,
            float new_track_thresh = 0.7,
            uint8_t track_buffer = 30,
            float match_thresh = 0.7,
            float proximity_thresh = 0.5,
            float appearance_thresh = 0.25,
            const char *gmc_method = "sparseOptFlow",
            uint8_t frame_rate = 30,
            float lambda = 0.985);
    ~BoTSORT() = default;

private:
    /**
     * @brief Extract visual features from the given frame and bounding box
     * 
     * @param frame Input frame
     * @param bbox_tlwh Bounding box (top, left, width, height)
     * @return FeatureVector Extracted visual features
     */
    FeatureVector _extract_features(const cv::Mat &frame, const cv::Rect_<float> &bbox_tlwh);

    /**
     * @brief Merge the given track lists
     * 
     * @param tracks_list_a Track list a
     * @param tracks_list_b Track list b
     * @return std::vector<std::shared_ptr<Track>> Merged track list
     */
    static std::vector<std::shared_ptr<Track>> _merge_track_lists(std::vector<std::shared_ptr<Track>> &tracks_list_a, std::vector<std::shared_ptr<Track>> &tracks_list_b);

    /**
     * @brief Remove tracks from the given track list
     * 
     * @param tracks_list List from which tracks are to be removed
     * @param tracks_to_remove Subset of tracks to be removed
     * @return std::vector<std::shared_ptr<Track>> Track list after removing tracks
     */
    static std::vector<std::shared_ptr<Track>> _remove_from_list(std::vector<std::shared_ptr<Track>> &tracks_list, std::vector<std::shared_ptr<Track>> &tracks_to_remove);

    /**
     * @brief Rectify track lists
     *  For any 2 tracks from lists a and b having IoU overlap < 0.15,
     *  the track with smaller history is considered as a false positive and removed
     * 
     * @param result_tracks_a Output track list a after rectification
     * @param result_tracks_b Output track list b after rectification
     * @param tracks_list_a Input track list a
     * @param tracks_list_b Input track list b
     */
    static void _remove_duplicate_tracks(
            std::vector<std::shared_ptr<Track>> &result_tracks_a,
            std::vector<std::shared_ptr<Track>> &result_tracks_b,
            std::vector<std::shared_ptr<Track>> &tracks_list_a,
            std::vector<std::shared_ptr<Track>> &tracks_list_b);
};
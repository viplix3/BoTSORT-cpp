#pragma once

#include "GlobalMotionCompensation.h"
#include "ReID.h"
#include "track.h"


#include <string>


class BoTSORT {
public:
    /**
     * @brief Execute the tracking algorithm on the given detections and frame
     * 
     * @param detections Detections (respresented using the Detection struct) in the current frame
     * @param frame Current frame
     * @return std::vector<Track>
     */
    std::vector<Track> track(const std::vector<Detection> &detections, const cv::Mat &frame);

private:
    bool _reid_enabled;
    uint8_t _track_buffer, _frame_rate, _frame_id, _buffer_size, _max_time_lost;
    float _track_high_thresh, _new_track_thresh, _match_thresh, _proximity_thresh, _appearance_thresh, _lambda;

    std::vector<Track *> _tracked_tracks;
    std::vector<Track *> _lost_tracks;
    std::vector<Track *> _removed_tracks;

    std::shared_ptr<KalmanFilter> _kalman_filter;
    std::unique_ptr<GlobalMotionCompensation> _gmc_algo;
    std::unique_ptr<ReIDModel> _reid_model;


public:
    BoTSORT(
            std::optional<const char *> model_weights = "",
            bool fp16_inference = false,
            float track_high_thresh = 0.45,
            float new_track_thresh = 0.6,
            uint8_t track_buffer = 30,
            float match_thresh = 0.8,
            float proximity_thresh = 0.5,
            float appearance_thresh = 0.25,
            const char *gmc_method = "sparseOptFlow",
            uint8_t frame_rate = 30,
            float lambda = 0.985);
    ~BoTSORT();

private:
    /**
     * @brief Extract visual features from the given bounding box using CNN
     * 
     * @param frame Image frame
     * @param bbox_tlwh Bounding box in the format (top left x, top left y, width, height)
     * @return FeatureVector Extracted visual features
     */
    FeatureVector _extract_features(const cv::Mat &frame, const cv::Rect_<float> &bbox_tlwh);

    /**
     * @brief Merge two track lists into one with no duplicates
     * 
     * @param tracks_list_a First track list
     * @param tracks_list_b Second track list
     * @return std::vector<Track *> Merged track list
     */
    std::vector<Track *> _merge_track_lists(std::vector<Track *> &tracks_list_a, std::vector<Track *> &tracks_list_b);

    /**
     * @brief Remove tracks from the given track list
     * 
     * @param tracks_list List from which tracks are to be removed
     * @param tracks_to_remove Subset of tracks to be removed
     * @return std::vector<Track *> List with tracks removed
     */
    std::vector<Track *> _remove_from_list(std::vector<Track *> &tracks_list, std::vector<Track *> &tracks_to_remove);

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
    void _remove_duplicate_tracks(
            std::vector<Track *> &result_tracks_a,
            std::vector<Track *> &result_tracks_b,
            std::vector<Track *> &tracks_list_a,
            std::vector<Track *> &tracks_list_b);
};
#pragma once

#include "KalmanFilter.h"
#include "KalmanFilterAccBased.h"
#include <deque>
#include <memory>

using KalmanFilter = bot_kalman::KalmanFilter;

enum TrackState {
    New = 0,
    Tracked,
    Lost,
    LongLost,
    Removed
};

class Track {
public:
    bool is_activated;
    int track_id;
    int state;

    uint32_t frame_id, tracklet_len, start_frame;

    std::vector<float> det_tlwh;
    std::shared_ptr<FeatureVector> curr_feat;
    std::unique_ptr<FeatureVector> smooth_feat;
    KFStateSpaceVec mean;
    KFStateSpaceMatrix covariance;

private:
    std::vector<float> _tlwh;
    std::vector<std::pair<uint8_t, float>> _class_hist;
    float _score;
    uint8_t _class_id;
    static constexpr float _alpha = 0.9;

    int _feat_history_size;
    std::deque<std::shared_ptr<FeatureVector>> _feat_history;


public:
    /**
     * @brief Construct a new Track object
     * 
     * @param tlwh Detection bounding box in the format [top-left-x, top-left-y, width, height]
     * @param score Detection score
     * @param class_id Detection class ID
     * @param feat (Optional) Detection feature vector
     * @param feat_history_size Size of the feature history (default: 50)
     */
    Track(std::vector<float> tlwh, float score, uint8_t class_id, std::optional<FeatureVector> feat = std::nullopt, int feat_history_size = 50);

    /**
     * @brief Get the next track ID
     * 
     * @return int Next track ID
     */
    static int next_id();

    /**
     * @brief Get end frame-id of the track
     * 
     * @return int End frame-id of the track
     */
    uint32_t end_frame() const;

    /**
     * @brief Upates the track state to Lost
     * 
     */
    void mark_lost();

    /**
     * @brief Upates the track state to LongLost
     * 
     */
    void mark_long_lost();

    /**
     * @brief Upates the track state to Removed
     * 
     */
    void mark_removed();

    /**
     * @brief Get the latest detection bounding box in the format [top-left-x, top-left-y, width, height]
     */
    std::vector<float> get_tlwh() const;

    /**
     * @brief Get the score object
     * 
     * @return float Confidence score of the track
     */
    float get_score() const;

    /**
     * @brief Activates the track
     * 
     * @param kalman_filter Kalman filter object for the track
     * @param frame_id Current frame-id
     */
    void activate(KalmanFilter &kalman_filter, uint32_t frame_id);

    /**
     * @brief Re-activates the track
     * 
     * @param kalman_filter Kalman filter object
     * @param new_track New track object
     * @param frame_id Current frame-id
     * @param new_id Whether to assign a new ID to the track (default: false)
     */
    void re_activate(KalmanFilter &kalman_filter, Track &new_track, uint32_t frame_id, bool new_id = false);

    /**
     * @brief Predict the next state of the track using the Kalman filter
     * 
     * @param kalman_filter Kalman filter class object
     */
    void predict(KalmanFilter &kalman_filter);

    /**
     * @brief Predict the next state of multiple tracks using the Kalman filter
     * 
     * @param tracks Tracks on which to perform the prediction step
     * @param kalman_filter Kalman filter object for the tracks
     */
    void static multi_predict(std::vector<std::shared_ptr<Track>> &tracks, KalmanFilter &kalman_filter);

    /**
     * @brief Apply camera motion to the track
     * 
     * @param H Homography matrix
     */
    void apply_camera_motion(const HomographyMatrix &H);

    /**
     * @brief Apply camera motion to multiple tracks
     * 
     * @param tracks Tracks on which to apply the camera motion
     * @param H Homography matrix
     */
    void static multi_gmc(std::vector<std::shared_ptr<Track>> &tracks, const HomographyMatrix &H);

    /**
     * @brief Update the track state using the new detection
     * 
     * @param new_track New track object to be used to update the old track
     * @param frame_id Current frame-id
     */
    void update(KalmanFilter &kalman_filter, Track &new_track, uint32_t frame_id);

private:
    /**
     * @brief Updates visual feature vector and feature history
     * Done by using a weighted average of the current feature vector and the previous feature vector
     * 
     * @param feat Current feature vector
     */
    void _update_features(const std::shared_ptr<FeatureVector> &feat);

    /**
     * @brief Populate a DetVec bbox object (xywh) from the detection bounding box (tlwh)
     * 
     * @param bbox_xywh DetVec bbox object (xywh) to be populated
     * @param tlwh Detection bounding box (tlwh)
     */
    static void _populate_DetVec_xywh(DetVec &bbox_xywh, const std::vector<float> &tlwh);

    /**
     * @brief Update the tracklet bounding box (stored as tlwh) inplace according to the tracker state
     * 
     */
    void _update_tracklet_tlwh_inplace();

    /**
     * @brief Update the class_id for the given tracklet on the basis of the frequency of the class_id in the class history
     * The history maintains a list of (class_id, score) pairs.
     * If a particular class_id occurs with a higher score, multiple times, it is more likely to be the correct class_id
     * If a particular class_id occurs with a lower score, multiple times, it is less likely to be the correct class_id
     * 
     * The frequency of the class_id is calculated as the sum of the scores of the class_id in the history
     * 
     * @param class_id Current class_id for the bounding box
     * @param score Current score for the bounding box
     */
    void _update_class_id(uint8_t class_id, float score);
};
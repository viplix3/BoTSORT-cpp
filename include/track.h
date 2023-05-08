#pragma once

#include "KalmanFilter.h"
#include <deque>

enum TrackState {
    New = 0,
    Tracked,
    Lost,
    LongLost,
    Removed
};

class Track {
public:
    /**
     * @brief Construct a new Track object
     * 
     * @param xywh Detection bounding box (xmid, ymid, width, height)
     * @param score Detection score
     * @param class_id Detection class ID
     * @param feat (Optional) Detection feature vector
     * @param feat_history_size Size of the feature history (default: 50)
     */
    Track(std::vector<float> xywh, float score, uint8_t class_id, std::optional<FeatureVector> feat = std::nullopt, int feat_history_size = 50);
    ~Track();

    /**
     * @brief Get the next track ID
     * 
     * @return int Next track ID
     */
    int next_id();

    /**
     * @brief Get end frame-id of the track
     * 
     * @return int End frame-id of the track
     */
    int end_frame();

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
     * @brief Activates the track
     * 
     * @param kalman_filter Kalman filter object for the track
     * @param frame_id Current frame-id
     */
    void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);


    void re_activate(Track &new_track, int frame_id, bool new_id = false);
    void static multi_predict(std::vector<Track *> &tracks, const byte_kalman::KalmanFilter &kalman_filter);
    void update(Track &new_track, int frame_id);

    bool is_activated;
    int track_id;
    int state;

    int frame_id;
    int tracklet_len;
    int start_frame;

    std::vector<float> det_xywh;

    KFStateSpaceVec mean;
    KFStateSpaceMatrix covariance;

private:
    std::vector<float> _tlwh;
    float _score;
    uint8_t _class_id;
    static constexpr float _alpha = 0.9;

    int _feat_history_size;
    FeatureVector _curr_feat, _smooth_feat;
    std::deque<FeatureVector> _feat_history;


    byte_kalman::KalmanFilter _kalman_filter;

    void _update_features(FeatureVector &feat);
};
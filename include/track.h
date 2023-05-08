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
    Track(std::vector<float> xywh, float score, uint8_t class_id, std::optional<FeatureVector> feat = std::nullopt, int feat_history_size = 50);
    ~Track();

    int next_id();
    int end_frame();
    void mark_lost();
    void mark_long_lost();
    void mark_removed();

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
#pragma once

#include "KalmanFilter.h"

enum TrackState {
    New = 0,
    Tracked,
    Lost,
    LongLost,
    Removed
};

class Track {
public:
    Track(std::vector<float> tlwh, float score, uint8_t class_id);
    ~Track();

    std::vector<float> static tlbr_to_tlwh(const std::vector<float> &tlbr);
    void static multi_predict(std::vector<Track *> &tracks, const byte_kalman::KalmanFilter &kalman_filter);

    void update_track_tlwh();

    std::vector<float> tlwh_to_xyah(std::vector<float> tlwh);

    void mark_lost();
    void mark_removed();
    int next_id();
    int end_frame();

    void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
    void re_activate(Track &new_track, int frame_id, bool new_id = false);
    void update(Track &new_track, int frame_id);

    bool is_activated;
    int track_id;
    int state;

    int frame_id;
    int tracklet_len;
    int start_frame;

    std::vector<float> det_tlwh;

    KFStateSpaceVec mean;
    KFStateSpaceMatrix covariance;

private:
    std::vector<float> _tlwh;
    float _score;
    uint8_t _class_id;

    byte_kalman::KalmanFilter kalman_filter;
};
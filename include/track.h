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
    Track(std::vector<float> tlwh_, float score);
    ~Track();

    std::vector<float> static tlbr_to_tlwh(const std::vector<float> &tlbr);
    void static multi_predict(std::vector<STrack *> &stracks, const byte_kalman::KalmanFilter &kalman_filter);
    void to_tlwh_inplace();
    void to_tlbr_inplace();

    std::vector<float> tlwh_to_xyah(std::vector<float> tlwh);

    void mark_lost();
    void mark_removed();
    int next_id();
    int end_frame();

    void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
    void re_activate(STrack &new_track, int frame_id, bool new_id = false);
    void update(STrack &new_track, int frame_id);

    bool is_activated;
    int track_id;
    int state;

    std::vector<float> _tlwh;
    std::vector<float> tlwh;
    int frame_id;
    int tracklet_len;
    int start_frame;

    KFStateSpaceVec mean;
    KFStateSpaceMatrix covariance;
    float score;
};
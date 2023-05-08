#include "track.h"

Track::Track(std::vector<float> xywh, float score, uint8_t class_id, std::optional<FeatureVector> feat, int feat_history_size) {
    // Save the detection in the det_xywh vector
    det_xywh.resize(DET_ELEMENTS);
    det_xywh.assign(xywh.begin(), xywh.end());

    _score = score;
    _class_id = class_id;

    tracklet_len = 0;
    is_activated = false;
    state = TrackState::New;

    if (feat) {
        _feat_history_size = feat_history_size;
        _update_features(*feat);
    }
}

Track::~Track() {
    // Nothing to do here
}

void Track::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id) {
    _kalman_filter = kalman_filter;
    track_id = next_id();

    // Create DetVec from det_xywh
    DetVec bbox_xywh;
    bbox_xywh << det_xywh[0], det_xywh[1], det_xywh[2], det_xywh[3];

    // Initialize the Kalman filter
    KFDataStateSpace state_space = _kalman_filter.init(bbox_xywh);
    mean = state_space.first;
    covariance = state_space.second;

    if (frame_id == 1) {
        is_activated = true;
    }
    this->frame_id = frame_id;
    start_frame = frame_id;
}

void Track::re_activate(Track &new_track, int frame_id, bool new_id = false) {
}


void Track::_update_features(FeatureVector &feat) {
    feat /= feat.norm();
    _curr_feat = feat;

    if (_feat_history.size() == 0) {
        _smooth_feat = feat;
    } else {
        _smooth_feat = _alpha * _smooth_feat + (1 - _alpha) * feat;
    }

    if (_feat_history.size() == _feat_history_size) {
        _feat_history.pop_front();
    }
    _feat_history.push_back(feat);
    _smooth_feat /= _smooth_feat.norm();
}

int Track::next_id() {
    static int _count = 0;
    _count++;
    return _count;
}

void Track::mark_lost() {
    state = TrackState::Lost;
}

void Track::mark_long_lost() {
    state = TrackState::LongLost;
}

void Track::mark_removed() {
    state = TrackState::Removed;
}

int Track::end_frame() {
    return frame_id;
}
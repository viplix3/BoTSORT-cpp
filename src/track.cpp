#include "track.h"

Track::Track(std::vector<float> tlwh, float score, uint8_t class_id, std::optional<FeatureVector> feat, int feat_history_size) {
    // Save the detection in the det_tlwh vector
    det_tlwh.resize(DET_ELEMENTS);
    det_tlwh.assign(tlwh.begin(), tlwh.end());

    _score = score;
    _class_id = class_id;

    tracklet_len = 0;
    is_activated = false;
    state = TrackState::New;

    if (feat) {
        _feat_history_size = feat_history_size;
        _update_features(*feat);
    }
    _update_tracklet_tlwh_inplace();
}

Track::~Track() {
    // Nothing to do here
}

void Track::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id) {
    _kalman_filter = kalman_filter;
    track_id = next_id();

    // Create DetVec from det_tlwh
    DetVec detection_bbox;
    _populate_DetVec_xywh(detection_bbox, det_tlwh);

    // Initialize the Kalman filter
    KFDataStateSpace state_space = _kalman_filter.init(detection_bbox);
    mean = state_space.first;
    covariance = state_space.second;

    if (frame_id == 1) {
        is_activated = true;
    }
    this->frame_id = frame_id;
    start_frame = frame_id;
    state = TrackState::Tracked;
    tracklet_len = 1;
    _update_tracklet_tlwh_inplace();
}

void Track::re_activate(Track &new_track, int frame_id, bool new_id) {
    DetVec new_track_bbox;
    _populate_DetVec_xywh(new_track_bbox, new_track._tlwh);

    KFDataStateSpace state_space = _kalman_filter.update(mean, covariance, new_track_bbox);
    mean = state_space.first;
    covariance = state_space.second;
    _update_tracklet_tlwh_inplace();
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

void Track::_populate_DetVec_xywh(DetVec &bbox_xywh, const std::vector<float> &tlwh) {
    bbox_xywh << tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2, tlwh[2], tlwh[3];
}

void Track::_update_tracklet_tlwh_inplace() {
    // If the tracklet is new, simply copy the det_tlwh
    if (state == TrackState::New) {
        _tlwh = det_tlwh;
        return;
    }

    // If the tracklet is not new, update the tlwh using the Kalman filter
    // mean. KF mean contains xywh, so need to convert
    _tlwh = {mean(0) - mean(2) / 2, mean(1) - mean(3) / 2, mean(2), mean(3)};
}
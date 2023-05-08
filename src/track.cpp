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
}

Track::~Track() {
    // Nothing to do here
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
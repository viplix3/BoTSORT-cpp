#include "track.h"

Track::Track(std::vector<float> tlwh, float score, uint8_t class_id, std::optional<FeatureVector> feat, int feat_history_size) {
    // Save the detection in the det_tlwh vector
    det_tlwh.resize(DET_ELEMENTS);
    det_tlwh.assign(tlwh.begin(), tlwh.end());

    _score = score;
    _class_id = -1;

    tracklet_len = 0;
    is_activated = false;
    state = TrackState::New;

    if (feat) {
        _feat_history_size = feat_history_size;
        _update_features(*feat);
    }

    _update_class_id(class_id, score);
    _update_tracklet_tlwh_inplace();
}

Track::~Track() {
    // Nothing to do here
}

void Track::activate(KalmanFilter &kalman_filter, int frame_id) {
    track_id = next_id();

    // Create DetVec from det_tlwh
    DetVec detection_bbox;
    _populate_DetVec_xywh(detection_bbox, det_tlwh);

    // Initialize the Kalman filter matrices
    KFDataStateSpace state_space = kalman_filter.init(detection_bbox);
    _mean = state_space.first;
    _covariance = state_space.second;

    if (frame_id == 1) {
        is_activated = true;
    }
    this->frame_id = frame_id;
    start_frame = frame_id;
    state = TrackState::Tracked;
    tracklet_len = 1;
    _update_tracklet_tlwh_inplace();
}

void Track::re_activate(KalmanFilter &kalman_filter, Track &new_track, int frame_id, bool new_id) {
    DetVec new_track_bbox;
    _populate_DetVec_xywh(new_track_bbox, new_track._tlwh);

    KFDataStateSpace state_space = kalman_filter.update(_mean, _covariance, new_track_bbox);
    _mean = state_space.first;
    _covariance = state_space.second;

    if (new_track._curr_feat.size() > 0) {
        _update_features(new_track._curr_feat);
    }

    if (new_id) {
        track_id = next_id();
    }

    tracklet_len = 0;
    state = TrackState::Tracked;
    is_activated = true;
    _score = new_track._score;
    this->frame_id = frame_id;

    _update_class_id(new_track._class_id, new_track._score);
    _update_tracklet_tlwh_inplace();
}

void Track::predict(KalmanFilter &kalman_filter) {
    kalman_filter.predict(_mean, _covariance);
    _update_tracklet_tlwh_inplace();
}

void Track::multi_predict(std::vector<Track *> &tracks, KalmanFilter &kalman_filter) {
    for (size_t i = 0; i < tracks.size(); i++) {
        tracks[i]->predict(kalman_filter);
    }
}

void Track::apply_camera_motion(const HomographyMatrix &H) {
    Eigen::MatrixXf R = H.block(0, 0, 2, 2);
    Eigen::VectorXf t = H.block(0, 2, 2, 1);

    Eigen::Matrix<float, 8, 8> R8x8 = Eigen::Matrix<float, 8, 8>::Identity();
    R8x8.block(0, 0, 2, 2) = R;

    _mean = R8x8 * _mean.transpose();
    _mean.head(2) += t;
    _covariance = R8x8 * _covariance * R8x8.transpose();
}


void Track::multi_cmc(std::vector<Track *> &tracks, const HomographyMatrix &H) {
    for (size_t i = 0; i < tracks.size(); i++) {
        tracks[i]->apply_camera_motion(H);
    }
}


void Track::update(KalmanFilter &kalman_filter, Track &new_track, int frame_id) {

    DetVec new_track_bbox;
    _populate_DetVec_xywh(new_track_bbox, new_track._tlwh);

    KFDataStateSpace state_space = kalman_filter.update(_mean, _covariance, new_track_bbox);

    if (new_track._curr_feat.size() > 0) {
        _update_features(new_track._curr_feat);
    }

    _mean = state_space.first;
    _covariance = state_space.second;
    state = TrackState::Tracked;
    is_activated = true;
    _score = new_track._score;
    tracklet_len++;
    this->frame_id = frame_id;

    _update_class_id(new_track._class_id, new_track._score);
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
    _tlwh = {_mean(0) - _mean(2) / 2, _mean(1) - _mean(3) / 2, _mean(2), _mean(3)};
}

void Track::_update_class_id(uint8_t class_id, float score) {
    if (_class_hist.size() > 0) {
        int max_freq = 0;
        bool found = false;

        for (auto &class_hist: _class_hist) {
            if (class_hist.first == class_id) {
                class_hist.second += score;
                found = true;
            }
            if (class_hist.second > max_freq) {
                max_freq = class_hist.second;
                _class_id = class_hist.first;
            }
        }

        if (!found) {
            _class_hist.push_back({class_id, score});
            _class_id = class_id;
        }
    } else {
        _class_hist.push_back({class_id, score});
        _class_id = class_id;
    }
}
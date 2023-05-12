#include "BoTSORT.h"
#include "matching.h"

BoTSORT::BoTSORT(
        std::optional<const char *> model_weights,
        bool fp16_inference,
        float track_high_thresh,
        float new_track_thresh,
        uint8_t track_buffer,
        float match_thresh,
        float proximity_thresh,
        float appearance_thresh,
        const char *gmc_method,
        uint8_t frame_rate,
        float lambda)
    : _track_high_thresh(track_high_thresh),
      _new_track_thresh(new_track_thresh),
      _track_buffer(track_buffer),
      _match_thresh(match_thresh),
      _proximity_thresh(proximity_thresh),
      _appearance_thresh(appearance_thresh),
      _frame_rate(frame_rate),
      _lambda(lambda) {

    // Tracker module
    _frame_id = 0;
    _buffer_size = static_cast<uint8_t>(_frame_rate / 30.0 * _track_buffer);
    _max_time_lost = _buffer_size;
    _kalman_filter = std::make_shared<KalmanFilter>(static_cast<double>(1.0 / _frame_rate));


    // Re-ID module, load visual feature extractor here
    if (model_weights.has_value()) {
        _reid_model = std::make_unique<ReIDModel>(model_weights.value(), fp16_inference);
    }


    // Global motion compensation module
    _gmc_algo = std::make_unique<GlobalMotionCompensation>(GMC_method_map[gmc_method]);
}

std::vector<Track> BoTSORT::track(const std::vector<Detection> &detections, const cv::Mat &frame) {
    ////////////////// Step 1: Create tracks for detections //////////////////
    // For all detections, extract features, create tracks and classify on the segregate of confidence
    _frame_id++;
    std::vector<Track *> detections_high_conf;
    std::vector<Track *> detections_low_conf;
    if (detections.size() > 0) {
        for (Detection &detection: const_cast<std::vector<Detection> &>(detections)) {
            detection.bbox_tlwh.x = std::max(0.0f, detection.bbox_tlwh.x);
            detection.bbox_tlwh.y = std::max(0.0f, detection.bbox_tlwh.y);
            detection.bbox_tlwh.width = std::min(static_cast<float>(frame.cols - 1), detection.bbox_tlwh.width);
            detection.bbox_tlwh.height = std::min(static_cast<float>(frame.rows - 1), detection.bbox_tlwh.height);

            FeatureVector embedding = _extract_features(frame, detection.bbox_tlwh);
            std::vector<float> tlwh = {detection.bbox_tlwh.x, detection.bbox_tlwh.y, detection.bbox_tlwh.width, detection.bbox_tlwh.height};
            Track track = Track(tlwh, detection.confidence, detection.class_id, embedding);

            if (detection.confidence >= _track_high_thresh) {
                detections_high_conf.push_back(&track);
            } else if (detection.confidence > 0.1 && detection.confidence < _track_high_thresh) {
                detections_low_conf.push_back(&track);
            }
        }
    }

    // Segregate tracks in unconfirmed and tracked tracks
    std::vector<Track *> unconfirmed_tracks, tracked_tracks;
    for (Track *track: _tracked_tracks) {
        if (!track->is_activated) {
            unconfirmed_tracks.push_back(track);
        } else {
            tracked_tracks.push_back(track);
        }
    }


    ////////////////// Step 2: First association, with high score detection boxes //////////////////
    // Merge currently tracked tracks and lost tracks
    std::vector<Track *> tracks_pool;
    tracks_pool = _merge_track_lists(tracked_tracks, _lost_tracks);

    // Predict the location of the tracks with KF (even for lost tracks)
    Track::multi_predict(tracks_pool, *_kalman_filter);

    // Estimate camera motion and apply camera motion compensation
    HomographyMatrix H = _gmc_algo->apply(frame, detections);
    Track::multi_gmc(tracks_pool, H);
    Track::multi_gmc(unconfirmed_tracks, H);

    // Associate tracks with high confidence detections
    CostMatrix raw_emd_dist = embedding_distance(tracks_pool, detections_high_conf);
    fuse_motion(*_kalman_filter, raw_emd_dist, tracks_pool, detections_high_conf, false, _lambda);

    // Added for code compilation
    return std::vector<Track>();
}

FeatureVector BoTSORT::_extract_features(const cv::Mat &frame, const cv::Rect_<float> &bbox_tlwh) {
    cv::Mat patch = frame(bbox_tlwh);
    cv::Mat patch_resized;
    return _reid_model->extract_features(patch_resized);
}

std::vector<Track *> BoTSORT::_merge_track_lists(std::vector<Track *> &tracks_list_a, std::vector<Track *> &tracks_list_b) {
    std::map<int, bool> exists;
    std::vector<Track *> merged_tracks_list;

    for (Track *track: tracks_list_a) {
        exists[track->track_id] = true;
        merged_tracks_list.push_back(track);
    }

    for (Track *track: tracks_list_b) {
        if (exists.find(track->track_id) == exists.end()) {
            exists[track->track_id] = true;
            merged_tracks_list.push_back(track);
        }
    }

    return merged_tracks_list;
}

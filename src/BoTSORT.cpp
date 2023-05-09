#include "BoTSORT.h"

BoTSORT::BoTSORT(
        std::string model_weights,
        bool fp16_inference,
        float track_high_thresh,
        float new_track_thresh,
        uint8_t track_buffer,
        float match_thresh,
        float proximity_thresh,
        float appearance_thresh,
        std::string gmc_method,
        uint8_t frame_rate,
        float lambda)
    : _model_weights(model_weights),
      _fp16_inference(fp16_inference),
      _track_high_thresh(track_high_thresh),
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
    _kalman_filter = byte_kalman::KalmanFilter(static_cast<double>(1.0 / _frame_rate));


    // Re-ID module, load visual feature extractor here
    auto _reid_model = ReIDModel(_model_weights, _fp16_inference);


    // Global motion compensation module
    _gmc_algo = GlobalMotionCompensation(GMC_Method_Map[gmc_method]);
}

BoTSORT::~BoTSORT() = default;

std::vector<Track> BoTSORT::track(const std::vector<Detection> &detections, const cv::Mat &frame) {
    ////////////////// Step 1: Create tracks for detections //////////////////
    _frame_id++;

    std::vector<Track> detections_high_conf;
    std::vector<Track> detections_low_conf;

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
                detections_high_conf.push_back(track);
            } else {
                detections_low_conf.push_back(track);
            }
        }
    }
}

FeatureVector BoTSORT::_extract_features(const cv::Mat &frame, const cv::Rect_<float> &bbox_tlwh) {
    cv::Mat patch = frame(bbox_tlwh);
    cv::Mat patch_resized;
    return _reid_model->extract_features(patch_resized);
}
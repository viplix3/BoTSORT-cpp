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


    // Re-ID module
    auto model = ReIDModel(_model_weights, _fp16_inference);// Load DL MODEL here


    // Global motion compensation module
    _gmc_algo = GlobalMotionCompensation(GMC_Method_Map[gmc_method]);
}

BoTSORT::~BoTSORT() = default;
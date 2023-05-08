#pragma once

#include "GlobalMotionCompensation.h"
#include "track.h"

#include <string>

struct Object {
    cv::Rect_<float> bbox;
    int class_id;
    float confidence;
};

class BoTSORT {
public:
    std::vector<Track> track(const std::vector<Object> &detections, const cv::Mat &frame);

private:
    GlobalMotionCompensation _gmc_algo;

    uint8_t _track_buffer, _frame_rate, _frame_id, _buffer_size, _max_time_lost;
    float _track_high_thresh, _new_track_thresh, _match_thresh, _proximity_thresh, _appearance_thresh, _lambda;

    bool _fp16_inference;
    std::string _model_weights;


    std::vector<Track> _tracked_tracks;
    std::vector<Track> _lost_tracks;
    std::vector<Track> _removed_tracks;

    byte_kalman::KalmanFilter _kalman_filter;


public:
    BoTSORT(
            std::string model_weights,
            bool fp16_inference,
            float track_high_thresh = 0.45,
            float new_track_thresh = 0.6,
            uint8_t track_buffer = 30,
            float match_thresh = 0.8,
            float proximity_thresh = 0.5,
            float appearance_thresh = 0.25,
            std::string gmc_method = "sparseOptFlow",
            uint8_t frame_rate = 30,
            float lambda = 0.985);
    ~BoTSORT();

private:
    std::vector<Track *> joint_tracks(std::vector<Track *> &tracks_list_a, std::vector<Track> &tracks_list_b);
    std::vector<Track> joint_tracks(std::vector<Track> &tracks_list_a, std::vector<Track> &tracks_list_b);

    std::vector<Track> sub_tracks(std::vector<Track> &tracks_list_a, std::vector<Track> &tracks_list_b);
    void remove_duplicate_tracks(
            std::vector<Track> &result_tracks_a,
            std::vector<Track> &result_tracks_b,
            std::vector<Track> &tracks_list_a,
            std::vector<Track> &tracks_list_b);
};
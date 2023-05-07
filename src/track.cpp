#include "track.h"

Track::Track(std::vector<float> tlwh, float score, uint8_t class_id) {
    // Save the detection in the det_tlwh vector
    det_tlwh.resize(DET_ELEMENTS);
    det_tlwh.assign(tlwh.begin(), tlwh.end());

    _score = score;
    _class_id = class_id;

    state = TrackState::New;
    is_activated = false;
    track_id = 0;

    frame_id = 0;
    tracklet_len = 0;
    start_frame = 0;
}
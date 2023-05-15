#include "BoTSORT.h"
#include "matching.h"
#include <unordered_set>

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
        _reid_enabled = true;
    } else {
        std::cout << "Re-ID module disabled" << std::endl;
        _reid_enabled = false;
    }


    // Global motion compensation module
    _gmc_algo = std::make_unique<GlobalMotionCompensation>(GlobalMotionCompensation::GMC_method_map[std::string(gmc_method)]);
}

std::vector<Track> BoTSORT::track(const std::vector<Detection> &detections, const cv::Mat &frame) {
    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////
    // For all detections, extract features, create tracks and classify on the segregate of confidence
    _frame_id++;
    std::vector<Track *> detections_high_conf, detections_low_conf;
    std::vector<Track *> activated_tracks, refind_tracks;

    if (detections.size() > 0) {
        for (Detection &detection: const_cast<std::vector<Detection> &>(detections)) {
            detection.bbox_tlwh.x = std::max(0.0f, detection.bbox_tlwh.x);
            detection.bbox_tlwh.y = std::max(0.0f, detection.bbox_tlwh.y);
            detection.bbox_tlwh.width = std::min(static_cast<float>(frame.cols - 1), detection.bbox_tlwh.width);
            detection.bbox_tlwh.height = std::min(static_cast<float>(frame.rows - 1), detection.bbox_tlwh.height);

            Track *tracklet;
            std::vector<float> tlwh = {detection.bbox_tlwh.x, detection.bbox_tlwh.y, detection.bbox_tlwh.width, detection.bbox_tlwh.height};
            if (_reid_enabled) {
                FeatureVector embedding = _extract_features(frame, detection.bbox_tlwh);
                tracklet = new Track(tlwh, detection.confidence, detection.class_id, embedding);
            } else {
                tracklet = new Track(tlwh, detection.confidence, detection.class_id);
            }

            if (detection.confidence >= _track_high_thresh) {
                detections_high_conf.push_back(tracklet);
            } else if (detection.confidence > 0.1 && detection.confidence < _track_high_thresh) {
                detections_low_conf.push_back(tracklet);
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
    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////


    ////////////////// Apply KF predict and GMC before running assocition algorithm //////////////////
    // Merge currently tracked tracks and lost tracks
    std::vector<Track *> tracks_pool;
    tracks_pool = _merge_track_lists(tracked_tracks, _lost_tracks);

    // Predict the location of the tracks with KF (even for lost tracks)
    Track::multi_predict(tracks_pool, *_kalman_filter);

    // Estimate camera motion and apply camera motion compensation
    HomographyMatrix H = _gmc_algo->apply(frame, detections);
    Track::multi_gmc(tracks_pool, H);
    Track::multi_gmc(unconfirmed_tracks, H);
    ////////////////// Apply KF predict and GMC before running assocition algorithm //////////////////


    ////////////////// ASSOCIATION ALGORITHM STARTS HERE //////////////////
    ////////////////// First association, with high score detection boxes //////////////////
    CostMatrix iou_dists, raw_emd_dist;

    // Find IoU distance between all tracked tracks and high confidence detections
    iou_dists = iou_distance(tracks_pool, detections_high_conf);
    fuse_score(iou_dists, detections_high_conf);// Fuse the score with IoU distance

    if (_reid_enabled) {
        // If re-ID is enabled, find the embedding distance between all tracked tracks and high confidence detections
        raw_emd_dist = embedding_distance(tracks_pool, detections_high_conf);
        fuse_motion(*_kalman_filter, raw_emd_dist, tracks_pool, detections_high_conf, false, _lambda);// Fuse the motion with embedding distance
    }

    // Fuse the IoU distance and embedding distance to get the final distance matrix
    CostMatrix distances_first_association = fuse_iou_with_emb(iou_dists, raw_emd_dist, _proximity_thresh, _appearance_thresh);

    // Perform linear assignment on the final distance matrix, LAPJV algorithm is used here
    AssociationData first_associations;
    linear_assignment(distances_first_association, _match_thresh, first_associations);

    // Update the tracks with the associated detections
    for (size_t i = 0; i < first_associations.matches.size(); i++) {
        Track *track = tracks_pool[first_associations.matches[i].first];
        Track *detection = detections_high_conf[first_associations.matches[i].second];

        // If track was being actively tracked, we update the track with the new associated detection
        if (track->state == TrackState::Tracked) {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
        } else {
            // If track was not being actively tracked, we re-activate the track with the new associated detection
            // NOTE: There should be a minimum number of frames before a track is re-activated
            track->re_activate(*_kalman_filter, *detection, _frame_id, false);
            refind_tracks.push_back(track);
        }
    }
    ////////////////// First association, with high score detection boxes //////////////////


    ////////////////// Second association, with low score detection boxes //////////////////
    // Get all unmatched but tracked tracks after the first association, these tracks will be used for the second association
    std::vector<Track *> unmatched_tracks_after_1st_association;
    for (size_t i = 0; i < first_associations.unmatched_track_indices.size(); i++) {
        int track_idx = first_associations.unmatched_track_indices[i];
        Track *track = tracks_pool[track_idx];
        if (track->state == TrackState::Tracked) {
            unmatched_tracks_after_1st_association.push_back(track);
        }
    }

    // Find IoU distance between unmatched but tracked tracks left after the first association and low confidence detections
    CostMatrix iou_dists_second;
    iou_dists_second = iou_distance(unmatched_tracks_after_1st_association, detections_low_conf);

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData second_associations;
    linear_assignment(iou_dists_second, 0.5, second_associations);

    // Update the tracks with the associated detections
    for (size_t i = 0; i < second_associations.matches.size(); i++) {
        Track *track = unmatched_tracks_after_1st_association[second_associations.matches[i].first];
        Track *detection = detections_low_conf[second_associations.matches[i].second];

        // If track was being actively tracked, we update the track with the new associated detection
        if (track->state == TrackState::Tracked) {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
        } else {
            // If track was not being actively tracked, we re-activate the track with the new associated detection
            // NOTE: There should be a minimum number of frames before a track is re-activated
            track->re_activate(*_kalman_filter, *detection, _frame_id, false);
            refind_tracks.push_back(track);
        }
    }

    // The tracks that are not associated with any detection even after the second association are marked as lost
    std::vector<Track *> lost_tracks;
    for (size_t i = 0; i < second_associations.unmatched_track_indices.size(); i++) {
        Track *track = unmatched_tracks_after_1st_association[second_associations.unmatched_track_indices[i]];
        if (track->state != TrackState::Lost) {
            track->mark_lost();
            lost_tracks.push_back(track);
        }
    }
    ////////////////// Second association, with low score detection boxes //////////////////


    ////////////////// Deal with unconfirmed tracks //////////////////
    std::vector<Track *> unmatched_detections_after_1st_association;
    for (size_t i = 0; i < first_associations.unmatched_det_indices.size(); i++) {
        int detection_idx = first_associations.unmatched_det_indices[i];
        Track *detection = detections_high_conf[detection_idx];
        unmatched_detections_after_1st_association.push_back(detection);
    }

    //Find IoU distance between unconfirmed tracks and high confidence detections left after the first association
    CostMatrix iou_dists_unconfirmed, raw_emd_dist_unconfirmed;
    iou_dists_unconfirmed = iou_distance(unconfirmed_tracks, unmatched_detections_after_1st_association);
    fuse_score(iou_dists_unconfirmed, unmatched_detections_after_1st_association);

    if (_reid_enabled) {
        // Find embedding distance between unconfirmed tracks and high confidence detections left after the first association
        raw_emd_dist_unconfirmed = embedding_distance(unconfirmed_tracks, unmatched_detections_after_1st_association);
        fuse_motion(*_kalman_filter, raw_emd_dist_unconfirmed, unconfirmed_tracks, unmatched_detections_after_1st_association, false, _lambda);
    }

    // Fuse the IoU distance and the embedding distance
    CostMatrix distances_unconfirmed = fuse_iou_with_emb(iou_dists_unconfirmed, raw_emd_dist_unconfirmed, _proximity_thresh, _appearance_thresh);

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData unconfirmed_associations;
    linear_assignment(distances_unconfirmed, 0.7, unconfirmed_associations);

    for (size_t i = 0; i < unconfirmed_associations.matches.size(); i++) {
        Track *track = unconfirmed_tracks[unconfirmed_associations.matches[i].first];
        Track *detection = unmatched_detections_after_1st_association[unconfirmed_associations.matches[i].second];

        // If the unconfrimed track is associated with a detection we update the track with the new associated detection
        // and add the track to the activated tracks list
        track->update(*_kalman_filter, *detection, _frame_id);
        activated_tracks.push_back(track);
    }

    // All the uncfonfirmed tracks that are not associated with any detection are marked as removed
    std::vector<Track *> removed_tracks;
    for (size_t i = 0; i < unconfirmed_associations.unmatched_track_indices.size(); i++) {
        Track *track = unconfirmed_tracks[unconfirmed_associations.unmatched_track_indices[i]];
        track->mark_removed();
        removed_tracks.push_back(track);
    }
    ////////////////// Deal with unconfirmed tracks //////////////////


    ////////////////// Initialize new tracks //////////////////
    std::vector<Track *> unmatched_high_conf_detections;
    for (size_t i = 0; i < unconfirmed_associations.unmatched_det_indices.size(); i++) {
        int detection_idx = unconfirmed_associations.unmatched_det_indices[i];
        Track *detection = unmatched_detections_after_1st_association[detection_idx];
        unmatched_high_conf_detections.push_back(detection);
    }

    // Initialize new tracks for the high confidence detections left after all the associations
    for (Track *detection: unmatched_high_conf_detections) {
        if (detection->get_score() >= _new_track_thresh) {
            detection->activate(*_kalman_filter, _frame_id);
            activated_tracks.push_back(detection);
        }
    }
    ////////////////// Initialize new tracks //////////////////


    ////////////////// Update lost tracks state //////////////////
    for (Track *track: _lost_tracks) {
        if (_frame_id - track->end_frame() > _max_time_lost) {
            track->mark_removed();
            removed_tracks.push_back(track);
        }
    }
    ////////////////// Update lost tracks state //////////////////


    ////////////////// Clean up the track lists //////////////////
    std::vector<Track *> upated_tracked_tracks;
    for (size_t i = 0; i < _tracked_tracks.size(); i++) {
        if (_tracked_tracks[i]->state == TrackState::Tracked) {
            upated_tracked_tracks.push_back(_tracked_tracks[i]);
        }
    }
    _tracked_tracks = _merge_track_lists(upated_tracked_tracks, activated_tracks);
    _tracked_tracks = _merge_track_lists(_tracked_tracks, refind_tracks);

    _lost_tracks = _merge_track_lists(_lost_tracks, lost_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, _tracked_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, _removed_tracks);
    _removed_tracks = _merge_track_lists(_removed_tracks, removed_tracks);

    std::vector<Track *> tracked_tracks_cleaned, lost_tracks_cleaned;
    _remove_duplicate_tracks(tracked_tracks_cleaned, lost_tracks_cleaned, _tracked_tracks, _lost_tracks);
    _tracked_tracks = tracked_tracks_cleaned, _lost_tracks = lost_tracks_cleaned;
    ////////////////// Clean up the track lists //////////////////


    ////////////////// Update output tracks //////////////////
    std::vector<Track> output_tracks;
    for (Track *track: _tracked_tracks) {
        if (track->is_activated) {
            output_tracks.push_back(*track);
        }
    }
    ////////////////// Update output tracks //////////////////

    return output_tracks;
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


std::vector<Track *> BoTSORT::_remove_from_list(std::vector<Track *> &tracks_list, std::vector<Track *> &tracks_to_remove) {
    std::map<int, bool> exists;
    std::vector<Track *> new_tracks_list;

    for (Track *track: tracks_to_remove) {
        exists[track->track_id] = true;
    }

    for (Track *track: tracks_list) {
        if (exists.find(track->track_id) == exists.end()) {
            new_tracks_list.push_back(track);
        }
    }

    return new_tracks_list;
}

void BoTSORT::_remove_duplicate_tracks(
        std::vector<Track *> &result_tracks_a,
        std::vector<Track *> &result_tracks_b,
        std::vector<Track *> &tracks_list_a,
        std::vector<Track *> &tracks_list_b) {
    CostMatrix iou_dists = iou_distance(tracks_list_a, tracks_list_b);

    std::unordered_set<size_t> dup_a, dup_b;
    for (size_t i = 0; i < iou_dists.rows(); i++) {
        for (size_t j = 0; j < iou_dists.cols(); j++) {
            if (iou_dists(i, j) < 0.15) {
                int time_a = tracks_list_a[i]->frame_id - tracks_list_a[i]->start_frame;
                int time_b = tracks_list_b[j]->frame_id - tracks_list_b[j]->start_frame;

                // We make an assumption that the longer trajectory is the correct one
                if (time_a > time_b) {
                    dup_b.insert(j);// In list b, track with index j is a duplicate
                } else {
                    dup_a.insert(i);// In list a, track with index i is a duplicate
                }
            }
        }
    }

    // Remove duplicates from the lists
    for (size_t i = 0; i < tracks_list_a.size(); i++) {
        if (dup_a.find(i) == dup_a.end()) {
            result_tracks_a.push_back(tracks_list_a[i]);
        }
    }

    for (size_t i = 0; i < tracks_list_b.size(); i++) {
        if (dup_b.find(i) == dup_b.end()) {
            result_tracks_b.push_back(tracks_list_b[i]);
        }
    }
}

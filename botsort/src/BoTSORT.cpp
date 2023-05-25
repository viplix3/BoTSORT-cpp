#include "BoTSORT.h"
#include "DataType.h"
#include "INIReader.h"
#include "matching.h"
#include <opencv2/imgproc.hpp>
#include <optional>
#include <unordered_set>

BoTSORT::BoTSORT(const std::string &config_dir) {
    _load_params_from_config(config_dir);

    // Tracker module
    _frame_id = 0;
    _buffer_size = static_cast<uint8_t>(_frame_rate / 30.0 * _track_buffer);
    _max_time_lost = _buffer_size;
    _kalman_filter = std::make_unique<KalmanFilter>(static_cast<double>(1.0 / _frame_rate));


    // Re-ID module, load visual feature extractor here
    if (_reid_model_weights_path) {
        _reid_model = std::make_unique<ReIDModel>(_reid_model_weights_path.value(), _fp16_inference);
        _reid_enabled = true;
    } else {
        std::cout << "Re-ID module disabled" << std::endl;
        _reid_enabled = false;
    }


    // Global motion compensation module
    _gmc_algo = std::make_unique<GlobalMotionCompensation>(GlobalMotionCompensation::GMC_method_map[_gmc_method_name], config_dir);
}


std::vector<std::shared_ptr<Track>> BoTSORT::track(const std::vector<Detection> &detections, const cv::Mat &frame) {
    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////
    // For all detections, extract features, create tracks and classify on the segregate of confidence
    _frame_id++;
    std::vector<std::shared_ptr<Track>> activated_tracks, refind_tracks;
    std::vector<std::shared_ptr<Track>> detections_high_conf, detections_low_conf;
    detections_low_conf.reserve(detections.size()), detections_high_conf.reserve(detections.size());

    if (!detections.empty()) {
        for (Detection &detection: const_cast<std::vector<Detection> &>(detections)) {
            detection.bbox_tlwh.x = std::max(0.0f, detection.bbox_tlwh.x);
            detection.bbox_tlwh.y = std::max(0.0f, detection.bbox_tlwh.y);
            detection.bbox_tlwh.width = std::min(static_cast<float>(frame.cols - 1), detection.bbox_tlwh.width);
            detection.bbox_tlwh.height = std::min(static_cast<float>(frame.rows - 1), detection.bbox_tlwh.height);

            std::shared_ptr<Track> tracklet;
            std::vector<float> tlwh = {detection.bbox_tlwh.x, detection.bbox_tlwh.y, detection.bbox_tlwh.width, detection.bbox_tlwh.height};

            if (detection.confidence > _track_low_thresh) {
                if (_reid_enabled) {
                    FeatureVector embedding = _extract_features(frame, detection.bbox_tlwh);
                    tracklet = std::make_shared<Track>(tlwh, detection.confidence, detection.class_id, embedding);
                } else {
                    tracklet = std::make_shared<Track>(tlwh, detection.confidence, detection.class_id);
                }

                if (detection.confidence >= _track_high_thresh) {
                    detections_high_conf.push_back(tracklet);
                } else {
                    detections_low_conf.push_back(tracklet);
                }
            }
        }
    }

    // Segregate tracks in unconfirmed and tracked tracks
    std::vector<std::shared_ptr<Track>> unconfirmed_tracks, tracked_tracks;
    for (const std::shared_ptr<Track> &track: _tracked_tracks) {
        if (!track->is_activated) {
            unconfirmed_tracks.push_back(track);
        } else {
            tracked_tracks.push_back(track);
        }
    }
    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////


    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////
    // Merge currently tracked tracks and lost tracks
    std::vector<std::shared_ptr<Track>> tracks_pool;
    tracks_pool = _merge_track_lists(tracked_tracks, _lost_tracks);

    // Predict the location of the tracks with KF (even for lost tracks)
    Track::multi_predict(tracks_pool, *_kalman_filter);

    // Estimate camera motion and apply camera motion compensation
    HomographyMatrix H = _gmc_algo->apply(frame, detections);
    Track::multi_gmc(tracks_pool, H);
    Track::multi_gmc(unconfirmed_tracks, H);
    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////


    ////////////////// ASSOCIATION ALGORITHM STARTS HERE //////////////////
    ////////////////// First association, with high score detection boxes //////////////////
    // Find IoU distance between all tracked tracks and high confidence detections
    CostMatrix iou_dists, raw_emd_dist, iou_dists_mask_1st_association, emd_dist_mask_1st_association;

    std::tie(iou_dists, iou_dists_mask_1st_association) = iou_distance(tracks_pool,
                                                                       detections_high_conf,
                                                                       _proximity_thresh);
    fuse_score(iou_dists, detections_high_conf);// Fuse the score with IoU distance

    if (_reid_enabled) {
        // If re-ID is enabled, find the embedding distance between all tracked tracks and high confidence detections
        std::tie(raw_emd_dist, emd_dist_mask_1st_association) = embedding_distance(tracks_pool,
                                                                                   detections_high_conf,
                                                                                   _appearance_thresh);
        fuse_motion(*_kalman_filter,
                    raw_emd_dist,
                    tracks_pool,
                    detections_high_conf,
                    _lambda);// Fuse the motion with embedding distance
    }

    // Fuse the IoU distance and embedding distance to get the final distance matrix
    CostMatrix distances_first_association = fuse_iou_with_emb(iou_dists,
                                                               raw_emd_dist,
                                                               iou_dists_mask_1st_association,
                                                               emd_dist_mask_1st_association);

    // Perform linear assignment on the final distance matrix, LAPJV algorithm is used here
    AssociationData first_associations = linear_assignment(distances_first_association, _match_thresh);

    // Update the tracks with the associated detections
    for (const std::pair<int, int> &match: first_associations.matches) {
        const std::shared_ptr<Track> &track = tracks_pool[match.first];
        const std::shared_ptr<Track> &detection = detections_high_conf[match.second];

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
    std::vector<std::shared_ptr<Track>> unmatched_tracks_after_1st_association;
    for (int track_idx: first_associations.unmatched_track_indices) {
        const std::shared_ptr<Track> &track = tracks_pool[track_idx];
        if (track->state == TrackState::Tracked) {
            unmatched_tracks_after_1st_association.push_back(track);
        }
    }

    // Find IoU distance between unmatched but tracked tracks left after the first association and low confidence detections
    CostMatrix iou_dists_second;
    iou_dists_second = iou_distance(unmatched_tracks_after_1st_association, detections_low_conf);

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData second_associations = linear_assignment(iou_dists_second, 0.5);

    // Update the tracks with the associated detections
    for (const std::pair<int, int> &match: second_associations.matches) {
        const std::shared_ptr<Track> &track = unmatched_tracks_after_1st_association[match.first];
        const std::shared_ptr<Track> &detection = detections_low_conf[match.second];

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
    std::vector<std::shared_ptr<Track>> lost_tracks;
    for (int unmatched_track_index: second_associations.unmatched_track_indices) {
        const std::shared_ptr<Track> &track = unmatched_tracks_after_1st_association[unmatched_track_index];
        if (track->state != TrackState::Lost) {
            track->mark_lost();
            lost_tracks.push_back(track);
        }
    }
    ////////////////// Second association, with low score detection boxes //////////////////


    ////////////////// Deal with unconfirmed tracks //////////////////
    std::vector<std::shared_ptr<Track>> unmatched_detections_after_1st_association;
    for (int detection_idx: first_associations.unmatched_det_indices) {
        const std::shared_ptr<Track> &detection = detections_high_conf[detection_idx];
        unmatched_detections_after_1st_association.push_back(detection);
    }

    //Find IoU distance between unconfirmed tracks and high confidence detections left after the first association
    CostMatrix iou_dists_unconfirmed, raw_emd_dist_unconfirmed, iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed;

    std::tie(iou_dists_unconfirmed, iou_dists_mask_unconfirmed) = iou_distance(unconfirmed_tracks,
                                                                               unmatched_detections_after_1st_association,
                                                                               _proximity_thresh);
    fuse_score(iou_dists_unconfirmed, unmatched_detections_after_1st_association);

    if (_reid_enabled) {
        // Find embedding distance between unconfirmed tracks and high confidence detections left after the first association
        std::tie(raw_emd_dist_unconfirmed, emd_dist_mask_unconfirmed) = embedding_distance(unconfirmed_tracks,
                                                                                           unmatched_detections_after_1st_association,
                                                                                           _appearance_thresh);
        fuse_motion(*_kalman_filter,
                    raw_emd_dist_unconfirmed,
                    unconfirmed_tracks,
                    unmatched_detections_after_1st_association,
                    _lambda);
    }

    // Fuse the IoU distance and the embedding distance
    CostMatrix distances_unconfirmed = fuse_iou_with_emb(iou_dists_unconfirmed,
                                                         raw_emd_dist_unconfirmed,
                                                         iou_dists_mask_unconfirmed,
                                                         emd_dist_mask_unconfirmed);

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData unconfirmed_associations = linear_assignment(distances_unconfirmed, 0.7);

    for (const std::pair<int, int> &match: unconfirmed_associations.matches) {
        const std::shared_ptr<Track> &track = unconfirmed_tracks[match.first];
        const std::shared_ptr<Track> &detection = unmatched_detections_after_1st_association[match.second];

        // If the unconfirmed track is associated with a detection we update the track with the new associated detection
        // and add the track to the activated tracks list
        track->update(*_kalman_filter, *detection, _frame_id);
        activated_tracks.push_back(track);
    }

    // All the unconfirmed tracks that are not associated with any detection are marked as removed
    std::vector<std::shared_ptr<Track>> removed_tracks;
    for (int unmatched_track_index: unconfirmed_associations.unmatched_track_indices) {
        const std::shared_ptr<Track> &track = unconfirmed_tracks[unmatched_track_index];
        track->mark_removed();
        removed_tracks.push_back(track);
    }
    ////////////////// Deal with unconfirmed tracks //////////////////


    ////////////////// Initialize new tracks //////////////////
    std::vector<std::shared_ptr<Track>> unmatched_high_conf_detections;
    for (int detection_idx: unconfirmed_associations.unmatched_det_indices) {
        const std::shared_ptr<Track> &detection = unmatched_detections_after_1st_association[detection_idx];
        unmatched_high_conf_detections.push_back(detection);
    }

    // Initialize new tracks for the high confidence detections left after all the associations
    for (const std::shared_ptr<Track> &detection: unmatched_high_conf_detections) {
        if (detection->get_score() >= _new_track_thresh) {
            detection->activate(*_kalman_filter, _frame_id);
            activated_tracks.push_back(detection);
        }
    }
    ////////////////// Initialize new tracks //////////////////


    ////////////////// Update lost tracks state //////////////////
    for (const std::shared_ptr<Track> &track: _lost_tracks) {
        if (_frame_id - track->end_frame() > _max_time_lost) {
            track->mark_removed();
            removed_tracks.push_back(track);
        }
    }
    ////////////////// Update lost tracks state //////////////////


    ////////////////// Clean up the track lists //////////////////
    std::vector<std::shared_ptr<Track>> updated_tracked_tracks;
    for (const std::shared_ptr<Track> &_tracked_track: _tracked_tracks) {
        if (_tracked_track->state == TrackState::Tracked) {
            updated_tracked_tracks.push_back(_tracked_track);
        }
    }
    _tracked_tracks = _merge_track_lists(updated_tracked_tracks, activated_tracks);
    _tracked_tracks = _merge_track_lists(_tracked_tracks, refind_tracks);

    _lost_tracks = _merge_track_lists(_lost_tracks, lost_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, _tracked_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, removed_tracks);

    std::vector<std::shared_ptr<Track>> tracked_tracks_cleaned, lost_tracks_cleaned;
    _remove_duplicate_tracks(tracked_tracks_cleaned, lost_tracks_cleaned, _tracked_tracks, _lost_tracks);
    _tracked_tracks = tracked_tracks_cleaned, _lost_tracks = lost_tracks_cleaned;
    ////////////////// Clean up the track lists //////////////////


    ////////////////// Update output tracks //////////////////
    std::vector<std::shared_ptr<Track>> output_tracks;
    for (const std::shared_ptr<Track> &track: _tracked_tracks) {
        if (track->is_activated) {
            output_tracks.push_back(track);
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

std::vector<std::shared_ptr<Track>> BoTSORT::_merge_track_lists(std::vector<std::shared_ptr<Track>> &tracks_list_a, std::vector<std::shared_ptr<Track>> &tracks_list_b) {
    std::map<int, bool> exists;
    std::vector<std::shared_ptr<Track>> merged_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_list_a) {
        exists[track->track_id] = true;
        merged_tracks_list.push_back(track);
    }

    for (const std::shared_ptr<Track> &track: tracks_list_b) {
        if (exists.find(track->track_id) == exists.end()) {
            exists[track->track_id] = true;
            merged_tracks_list.push_back(track);
        }
    }

    return merged_tracks_list;
}


std::vector<std::shared_ptr<Track>> BoTSORT::_remove_from_list(std::vector<std::shared_ptr<Track>> &tracks_list, std::vector<std::shared_ptr<Track>> &tracks_to_remove) {
    std::map<int, bool> exists;
    std::vector<std::shared_ptr<Track>> new_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_to_remove) {
        exists[track->track_id] = true;
    }

    for (const std::shared_ptr<Track> &track: tracks_list) {
        if (exists.find(track->track_id) == exists.end()) {
            new_tracks_list.push_back(track);
        }
    }

    return new_tracks_list;
}

void BoTSORT::_remove_duplicate_tracks(
        std::vector<std::shared_ptr<Track>> &result_tracks_a,
        std::vector<std::shared_ptr<Track>> &result_tracks_b,
        std::vector<std::shared_ptr<Track>> &tracks_list_a,
        std::vector<std::shared_ptr<Track>> &tracks_list_b) {
    CostMatrix iou_dists = iou_distance(tracks_list_a, tracks_list_b);

    std::unordered_set<size_t> dup_a, dup_b;
    for (Eigen::Index i = 0; i < iou_dists.rows(); i++) {
        for (Eigen::Index j = 0; j < iou_dists.cols(); j++) {
            if (iou_dists(i, j) < 0.15) {
                int time_a = static_cast<int>(tracks_list_a[i]->frame_id - tracks_list_a[i]->start_frame);
                int time_b = static_cast<int>(tracks_list_b[j]->frame_id - tracks_list_b[j]->start_frame);

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


void BoTSORT::_load_params_from_config(const std::string &config_dir) {
    const std::string tracker_name = "BoTSORT";

    INIReader tracker_config(config_dir + "/tracker.ini");
    if (tracker_config.ParseError() < 0) {
        std::cout << "Can't load " << config_dir << "/tracker.ini" << std::endl;
        exit(1);
    }

    _reid_model_weights_path = tracker_config.Get(tracker_name, "model_path");
    _fp16_inference = tracker_config.GetBoolean(tracker_name, "fp16_inference", false);

    _track_high_thresh = tracker_config.GetFloat(tracker_name, "track_high_thresh", 0.6F);
    _track_low_thresh = tracker_config.GetFloat(tracker_name, "track_low_thresh", 0.1F);
    _new_track_thresh = tracker_config.GetFloat(tracker_name, "new_track_thresh", 0.7F);

    _track_buffer = tracker_config.GetInteger(tracker_name, "track_buffer", 30);

    _match_thresh = tracker_config.GetFloat(tracker_name, "match_thresh", 0.7F);
    _proximity_thresh = tracker_config.GetFloat(tracker_name, "proximity_thresh", 0.5F);
    _appearance_thresh = tracker_config.GetFloat(tracker_name, "appearance_thresh", 0.25F);

    _gmc_method_name = tracker_config.Get(tracker_name, "gmc_method", "sparseOptFlow");

    _frame_rate = tracker_config.GetInteger(tracker_name, "frame_rate", 30);
    _lambda = tracker_config.GetFloat(tracker_name, "lambda", 0.985F);
}
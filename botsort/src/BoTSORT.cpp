#include "BoTSORT.h"

#include <optional>
#include <unordered_set>

#include <opencv2/imgproc.hpp>

#include "DataType.h"
#include "INIReader.h"
#include "matching.h"
#include "profiler.h"

namespace
{
template<typename T>
bool requires_load(const Config<T> &config)
{
    return std::holds_alternative<std::string>(config) &&
           !std::get<std::string>(config).empty();
}

template<typename T>
bool not_empty(const Config<T> &config)
{
    bool has_config = !std::holds_alternative<std::monostate>(config);
    bool is_non_empty = !(std::holds_alternative<std::string>(config) &&
                          std::get<std::string>(config).empty());

    return has_config && is_non_empty;
}

template<typename T>
T fetch_config(const Config<T> &config,
               std::function<T(const std::string &)> loader)
{
    if (std::holds_alternative<T>(config))
    {
        return std::get<T>(config);
    }

    if (requires_load(config))
    {
        return loader(std::get<std::string>(config));
    }

    throw std::runtime_error("Config is empty");
}
}// namespace

BoTSORT::BoTSORT(const Config<TrackerParams> &tracker_config,
                 const Config<GMC_Params> &gmc_config,
                 const Config<ReIDParams> &reid_config,
                 const std::string &reid_onnx_model_path,
                 bool pitch_kalman_enabled,
                 float pitch_kf_std_weight_position_m,
                 float pitch_kf_std_weight_velocity_m)
{
    auto tracker_params = fetch_config<TrackerParams>(
            tracker_config, TrackerParams::load_config);
    _load_params_from_config(tracker_params);

    // Tracker module
    _frame_id = 0;
    _buffer_size = static_cast<uint8_t>(_frame_rate / 30.0 * _track_buffer);
    _max_time_lost = _buffer_size;
    _kalman_filter = std::make_unique<KalmanFilter>(
            static_cast<double>(1.0 / _frame_rate));

    // Phase E.2 — metre-space Kalman, optional. When disabled the rest of
    // _track_impl bypasses every metre code path (see pitch_pass_active),
    // making behavior bit-identical to pre-E.2.
    _pitch_kalman_enabled = pitch_kalman_enabled;
    if (_pitch_kalman_enabled) {
        const float dt = 1.0F / static_cast<float>(_frame_rate);
        _pitch_kalman_filter = std::make_unique<PitchKalmanFilter>(
                dt, pitch_kf_std_weight_position_m,
                pitch_kf_std_weight_velocity_m);
    }


    // Re-ID module, load visual feature extractor here
    if (_reid_enabled && not_empty(reid_config) &&
        reid_onnx_model_path.size() > 0)
    {
        auto reid_params =
                fetch_config<ReIDParams>(reid_config, ReIDParams::load_config);
        _reid_model =
                std::make_unique<ReIDModel>(reid_params, reid_onnx_model_path);
    }
    else
    {
        std::cout << "Re-ID module disabled" << std::endl;
        _reid_enabled = false;
    }

    // Cache the distance metric. If a ReID model was loaded, defer to it;
    // otherwise default to "cosine" so callers supplying externally-computed
    // embeddings (via the track-with-features overload) still get a sensible
    // metric without needing the internal model.
    _distance_metric =
            _reid_model ? _reid_model->get_distance_metric() : "cosine";

    // Global motion compensation module
    if (_gmc_enabled && not_empty(gmc_config))
    {
        auto gmc_params = fetch_config<
                GMC_Params>(gmc_config, [this](const std::string &config_path) {
            return GMC_Params::load_config(
                    GlobalMotionCompensation::GMC_method_map[_gmc_method_name],
                    config_path);
        });
        _gmc_algo = std::make_unique<GlobalMotionCompensation>(gmc_params);
    }
    else
    {
        std::cout << "GMC disabled" << std::endl;
        _gmc_enabled = false;
    }
}


std::vector<std::shared_ptr<Track>>
BoTSORT::track(const std::vector<Detection> &detections, const cv::Mat &frame)
{
    // Delegate to the features-aware overload with an empty feature vector;
    // it preserves the legacy behaviour (internal ReID extraction when
    // _reid_enabled, motion-only otherwise).
    return track(detections, std::vector<FeatureVector>{}, frame);
}


std::vector<std::shared_ptr<Track>>
BoTSORT::track(const std::vector<Detection> &detections,
               const std::vector<FeatureVector> &features,
               const cv::Mat &frame)
{
    return _track_impl(detections, features, frame, nullptr);
}


std::vector<std::shared_ptr<Track>>
BoTSORT::track(const std::vector<Detection> &detections,
               const std::vector<FeatureVector> &features,
               const cv::Mat &frame, const HomographyMatrix &H)
{
    return _track_impl(detections, features, frame, &H);
}


std::vector<std::shared_ptr<Track>>
BoTSORT::track(const std::vector<Detection> &detections,
               const std::vector<FeatureVector> &features,
               const cv::Mat &frame, const HomographyMatrix &H,
               const std::vector<std::optional<bot_kalman::PKFMeasVec>>&
                       metre_measurements)
{
    // Validate parallel-array contract. An empty metre_measurements vector
    // means "metre path off this frame" and is always permitted; a non-empty
    // vector must be 1:1 with detections (same indexing).
    if (!metre_measurements.empty() &&
        metre_measurements.size() != detections.size())
    {
        throw std::invalid_argument(
                "BoTSORT::track — metre_measurements size mismatch with "
                "detections");
    }
    // Stash for the duration of this _track_impl call. We use a raw pointer
    // because std::vector<std::optional<...>> isn't trivially copyable on
    // the hot path; an RAII guard clears it on ALL exit paths (including
    // exceptions from _track_impl) so subsequent track() calls that never
    // set the pointer don't dereference a popped stack frame.
    struct MetreStashGuard
    {
        BoTSORT *self;
        ~MetreStashGuard() { self->_frame_metre_measurements = nullptr; }
    };
    _frame_metre_measurements = &metre_measurements;
    MetreStashGuard guard{this};
    return _track_impl(detections, features, frame, &H);
}


std::vector<std::shared_ptr<Track>>
BoTSORT::_track_impl(const std::vector<Detection> &detections,
                     const std::vector<FeatureVector> &features,
                     const cv::Mat &frame,
                     const HomographyMatrix *precomputed_H)
{
    PROFILE_FUNCTION();

    // Validate parallel-array contract. Empty features is allowed and means
    // "fall back to internal extraction or motion-only".
    if (!features.empty() && features.size() != detections.size())
    {
        throw std::invalid_argument(
                "BoTSORT::track: features.size() must equal "
                "detections.size() when features are supplied");
    }

    // Appearance-aware association is enabled when either (a) the user has
    // supplied externally-computed embeddings, or (b) the legacy internal
    // ReID path is active. The TrackerParams::reid_enabled flag retains its
    // original meaning: it gates internal extraction only.
    const bool external_features = !features.empty();
    const bool use_appearance = external_features || _reid_enabled;

    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////
    // For all detections, extract features, create tracks and classify on the segregate of confidence
    _frame_id++;
    std::vector<std::shared_ptr<Track>> activated_tracks, refind_tracks;
    std::vector<std::shared_ptr<Track>> detections_high_conf,
            detections_low_conf;
    detections_low_conf.reserve(detections.size()),
            detections_high_conf.reserve(detections.size());

    // Phase E.2 — parallel metre-measurement subsets that match the
    // high-conf / low-conf detection vectors index-for-index. We populate
    // them in the same pass as the detection split so the column ordering
    // stays in lock-step with the cost matrices later on.
    const bool have_metre = (_frame_metre_measurements != nullptr) &&
                            !_frame_metre_measurements->empty();
    std::vector<std::optional<bot_kalman::PKFMeasVec>>
            metre_meas_high_conf, metre_meas_low_conf;
    if (have_metre) {
        metre_meas_high_conf.reserve(detections.size());
        metre_meas_low_conf.reserve(detections.size());
    }

    if (!detections.empty())
    {
        for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx)
        {
            Detection &detection =
                    const_cast<Detection &>(detections[det_idx]);
            detection.bbox_tlwh.x = std::max(0.0f, detection.bbox_tlwh.x);
            detection.bbox_tlwh.y = std::max(0.0f, detection.bbox_tlwh.y);
            detection.bbox_tlwh.width =
                    std::min(static_cast<float>(frame.cols - 1),
                             detection.bbox_tlwh.width);
            detection.bbox_tlwh.height =
                    std::min(static_cast<float>(frame.rows - 1),
                             detection.bbox_tlwh.height);

            std::shared_ptr<Track> tracklet;
            std::vector<float> tlwh = {
                    detection.bbox_tlwh.x, detection.bbox_tlwh.y,
                    detection.bbox_tlwh.width, detection.bbox_tlwh.height};

            if (detection.confidence > _track_low_thresh)
            {
                if (external_features)
                {
                    // Use the caller-supplied embedding verbatim — no host
                    // round-trip through cv::Mat, no internal model needed.
                    tracklet = std::make_shared<Track>(
                            tlwh, detection.confidence, detection.class_id,
                            features[det_idx]);
                }
                else if (_reid_enabled)
                {
                    FeatureVector embedding =
                            _extract_features(frame, detection.bbox_tlwh);
                    tracklet = std::make_shared<Track>(
                            tlwh, detection.confidence, detection.class_id,
                            embedding);
                }
                else
                    tracklet = std::make_shared<Track>(
                            tlwh, detection.confidence, detection.class_id);

                if (detection.confidence >= _track_high_thresh)
                {
                    detections_high_conf.push_back(tracklet);
                    if (have_metre) {
                        metre_meas_high_conf.push_back(
                                (*_frame_metre_measurements)[det_idx]);
                    }
                }
                else
                {
                    detections_low_conf.push_back(tracklet);
                    if (have_metre) {
                        metre_meas_low_conf.push_back(
                                (*_frame_metre_measurements)[det_idx]);
                    }
                }
            }
        }
    }

    // Segregate tracks in unconfirmed and tracked tracks
    std::vector<std::shared_ptr<Track>> unconfirmed_tracks, tracked_tracks;
    for (const std::shared_ptr<Track> &track: _tracked_tracks)
    {
        if (!track->is_activated)
        {
            unconfirmed_tracks.push_back(track);
        }
        else
        {
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

    // Phase E.2 — predict metre-space state for tracks that have it initialized.
    // Skipped entirely when pitch_kalman is disabled OR when this frame has no
    // metre measurements (host's homography invalid). The bool is computed
    // here once and reused by every subsequent metre code path so the
    // disabled path is bit-identical to pre-E.2.
    const bool pitch_pass_active = _pitch_kalman_enabled &&
                                   _pitch_kalman_filter &&
                                   have_metre;
    if (pitch_pass_active) {
        for (auto& track_ptr : tracks_pool) {
            if (track_ptr->pitch_kf_initialized()) {
                track_ptr->predict_pitch(*_pitch_kalman_filter);
            }
        }
    }

    // Apply camera motion compensation. When the caller supplied a homography
    // (e.g. one upstream GMC pass shared by multiple BoTSORT instances), use
    // it verbatim and skip the internal _gmc_algo->apply() call entirely.
    // Otherwise honour _gmc_enabled and compute it ourselves.
    if (precomputed_H != nullptr)
    {
        Track::multi_gmc(tracks_pool, *precomputed_H);
        Track::multi_gmc(unconfirmed_tracks, *precomputed_H);
    }
    else if (_gmc_enabled)
    {
        HomographyMatrix H = _gmc_algo->apply(frame, detections);
        Track::multi_gmc(tracks_pool, H);
        Track::multi_gmc(unconfirmed_tracks, H);
    }
    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////


    ////////////////// ASSOCIATION ALGORITHM STARTS HERE //////////////////
    ////////////////// First association, with high score detection boxes //////////////////
    // Find IoU distance between all tracked tracks and high confidence detections
    CostMatrix iou_dists, raw_emd_dist, iou_dists_mask_1st_association,
            emd_dist_mask_1st_association;

    std::tie(iou_dists, iou_dists_mask_1st_association) =
            iou_distance(tracks_pool, detections_high_conf, _proximity_thresh);
    fuse_score(iou_dists,
               detections_high_conf);// Fuse the score with IoU distance

    if (use_appearance)
    {
        // If appearance-aware association is enabled (either via internal
        // ReID or externally-supplied embeddings), find the embedding
        // distance between all tracked tracks and high confidence detections
        std::tie(raw_emd_dist, emd_dist_mask_1st_association) =
                embedding_distance(tracks_pool, detections_high_conf,
                                   _appearance_thresh, _distance_metric);
        fuse_motion(*_kalman_filter, raw_emd_dist, tracks_pool,
                    detections_high_conf,
                    _lambda);// Fuse the motion with embedding distance
    }

    if (pitch_pass_active) {
        // Phase E.2 — additive metre Mahalanobis tightening on the cost
        // matrices feeding the 1st association. We tighten BOTH the
        // appearance-side matrix (only relevant when use_appearance) and the
        // IoU matrix so the metre gate isn't bypassed for (track, det) pairs
        // that the appearance branch never visits.
        if (use_appearance) {
            fuse_motion_pitch(*_pitch_kalman_filter, raw_emd_dist,
                              tracks_pool, detections_high_conf,
                              metre_meas_high_conf);
        }
        fuse_motion_pitch(*_pitch_kalman_filter, iou_dists, tracks_pool,
                          detections_high_conf, metre_meas_high_conf);
    }

    // Fuse the IoU distance and embedding distance to get the final distance matrix
    CostMatrix distances_first_association = fuse_iou_with_emb(
            iou_dists, raw_emd_dist, iou_dists_mask_1st_association,
            emd_dist_mask_1st_association);

    // Perform linear assignment on the final distance matrix, LAPJV algorithm is used here
    AssociationData first_associations =
            linear_assignment(distances_first_association, _match_thresh);

    // Update the tracks with the associated detections
    for (const std::pair<int, int> &match: first_associations.matches)
    {
        const std::shared_ptr<Track> &track = tracks_pool[match.first];
        const std::shared_ptr<Track> &detection =
                detections_high_conf[match.second];

        // If track was being actively tracked, we update the track with the new associated detection
        if (track->state == TrackState::Tracked)
        {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
        }
        else
        {
            // If track was not being actively tracked, we re-activate the track with the new associated detection
            // NOTE: There should be a minimum number of frames before a track is re-activated
            track->re_activate(*_kalman_filter, *detection, _frame_id, false);
            refind_tracks.push_back(track);
        }

        // Phase E.2 — keep metre-space state in lock-step with the pixel KF.
        // update_pitch lazily activates on first use, so no explicit
        // activate_pitch branch needed for re-activations either.
        if (pitch_pass_active) {
            const auto& maybe_metre = metre_meas_high_conf[match.second];
            if (maybe_metre.has_value()) {
                track->update_pitch(*_pitch_kalman_filter, maybe_metre.value());
            }
        }
    }
    ////////////////// First association, with high score detection boxes //////////////////


    ////////////////// Second association, with low score detection boxes //////////////////
    // Get all unmatched but tracked tracks after the first association, these tracks will be used for the second association
    std::vector<std::shared_ptr<Track>> unmatched_tracks_after_1st_association;
    for (int track_idx: first_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track = tracks_pool[track_idx];
        if (track->state == TrackState::Tracked)
        {
            unmatched_tracks_after_1st_association.push_back(track);
        }
    }

    // Find IoU distance between unmatched but tracked tracks left after the first association and low confidence detections
    CostMatrix iou_dists_second;
    iou_dists_second = iou_distance(unmatched_tracks_after_1st_association,
                                    detections_low_conf);

    if (pitch_pass_active) {
        // Phase E.2 — tighten the 2nd-association IoU matrix with the metre
        // gate. The 2nd association doesn't run fuse_motion at all (pixel KF
        // gating is skipped here in the original code); metre gate is the
        // only Mahalanobis check on this branch.
        fuse_motion_pitch(*_pitch_kalman_filter, iou_dists_second,
                          unmatched_tracks_after_1st_association,
                          detections_low_conf, metre_meas_low_conf);
    }

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData second_associations =
            linear_assignment(iou_dists_second, 0.5);

    // Update the tracks with the associated detections
    for (const std::pair<int, int> &match: second_associations.matches)
    {
        const std::shared_ptr<Track> &track =
                unmatched_tracks_after_1st_association[match.first];
        const std::shared_ptr<Track> &detection =
                detections_low_conf[match.second];

        // If track was being actively tracked, we update the track with the new associated detection
        if (track->state == TrackState::Tracked)
        {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
        }
        else
        {
            // If track was not being actively tracked, we re-activate the track with the new associated detection
            // NOTE: There should be a minimum number of frames before a track is re-activated
            track->re_activate(*_kalman_filter, *detection, _frame_id, false);
            refind_tracks.push_back(track);
        }

        // Phase E.2 — metre update for the 2nd-association branch.
        if (pitch_pass_active) {
            const auto& maybe_metre = metre_meas_low_conf[match.second];
            if (maybe_metre.has_value()) {
                track->update_pitch(*_pitch_kalman_filter, maybe_metre.value());
            }
        }
    }

    // The tracks that are not associated with any detection even after the second association are marked as lost
    std::vector<std::shared_ptr<Track>> lost_tracks;
    for (int unmatched_track_index: second_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track =
                unmatched_tracks_after_1st_association[unmatched_track_index];
        if (track->state != TrackState::Lost)
        {
            track->mark_lost();
            lost_tracks.push_back(track);
        }
    }
    ////////////////// Second association, with low score detection boxes //////////////////


    ////////////////// Deal with unconfirmed tracks //////////////////
    std::vector<std::shared_ptr<Track>>
            unmatched_detections_after_1st_association;
    // Phase E.2 — parallel metre vec for the unconfirmed pass; sourced from
    // metre_meas_high_conf at the same indices used to build the detection
    // subset above so column ordering matches the cost matrices.
    std::vector<std::optional<bot_kalman::PKFMeasVec>>
            metre_meas_unmatched_high_conf;
    if (pitch_pass_active) {
        metre_meas_unmatched_high_conf.reserve(
                first_associations.unmatched_det_indices.size());
    }
    for (int detection_idx: first_associations.unmatched_det_indices)
    {
        const std::shared_ptr<Track> &detection =
                detections_high_conf[detection_idx];
        unmatched_detections_after_1st_association.push_back(detection);
        if (pitch_pass_active) {
            metre_meas_unmatched_high_conf.push_back(
                    metre_meas_high_conf[detection_idx]);
        }
    }

    //Find IoU distance between unconfirmed tracks and high confidence detections left after the first association
    CostMatrix iou_dists_unconfirmed, raw_emd_dist_unconfirmed,
            iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed;

    std::tie(iou_dists_unconfirmed, iou_dists_mask_unconfirmed) = iou_distance(
            unconfirmed_tracks, unmatched_detections_after_1st_association,
            _proximity_thresh);
    fuse_score(iou_dists_unconfirmed,
               unmatched_detections_after_1st_association);

    if (use_appearance)
    {
        // Find embedding distance between unconfirmed tracks and high confidence detections left after the first association
        std::tie(raw_emd_dist_unconfirmed, emd_dist_mask_unconfirmed) =
                embedding_distance(unconfirmed_tracks,
                                   unmatched_detections_after_1st_association,
                                   _appearance_thresh, _distance_metric);
        fuse_motion(*_kalman_filter, raw_emd_dist_unconfirmed,
                    unconfirmed_tracks,
                    unmatched_detections_after_1st_association, _lambda);
    }

    if (pitch_pass_active) {
        // Phase E.2 — additive metre Mahalanobis tightening on the
        // unconfirmed-tracks association. Same rationale as the 1st
        // association: tighten both the appearance (when present) and
        // IoU matrices.
        if (use_appearance) {
            fuse_motion_pitch(*_pitch_kalman_filter, raw_emd_dist_unconfirmed,
                              unconfirmed_tracks,
                              unmatched_detections_after_1st_association,
                              metre_meas_unmatched_high_conf);
        }
        fuse_motion_pitch(*_pitch_kalman_filter, iou_dists_unconfirmed,
                          unconfirmed_tracks,
                          unmatched_detections_after_1st_association,
                          metre_meas_unmatched_high_conf);
    }

    // Fuse the IoU distance and the embedding distance
    CostMatrix distances_unconfirmed = fuse_iou_with_emb(
            iou_dists_unconfirmed, raw_emd_dist_unconfirmed,
            iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed);

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData unconfirmed_associations =
            linear_assignment(distances_unconfirmed, 0.7);

    for (const std::pair<int, int> &match: unconfirmed_associations.matches)
    {
        const std::shared_ptr<Track> &track = unconfirmed_tracks[match.first];
        const std::shared_ptr<Track> &detection =
                unmatched_detections_after_1st_association[match.second];

        // If the unconfirmed track is associated with a detection we update the track with the new associated detection
        // and add the track to the activated tracks list
        track->update(*_kalman_filter, *detection, _frame_id);
        activated_tracks.push_back(track);

        // Phase E.2 — metre update for the unconfirmed-track branch.
        if (pitch_pass_active) {
            const auto& maybe_metre =
                    metre_meas_unmatched_high_conf[match.second];
            if (maybe_metre.has_value()) {
                track->update_pitch(*_pitch_kalman_filter, maybe_metre.value());
            }
        }
    }

    // All the unconfirmed tracks that are not associated with any detection are marked as removed
    std::vector<std::shared_ptr<Track>> removed_tracks;
    for (int unmatched_track_index:
         unconfirmed_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track =
                unconfirmed_tracks[unmatched_track_index];
        track->mark_removed();
        removed_tracks.push_back(track);
    }
    ////////////////// Deal with unconfirmed tracks //////////////////


    ////////////////// Initialize new tracks //////////////////
    std::vector<std::shared_ptr<Track>> unmatched_high_conf_detections;
    // Phase E.2 — parallel metre vec for newly-born tracks.
    std::vector<std::optional<bot_kalman::PKFMeasVec>>
            metre_meas_new_tracks;
    if (pitch_pass_active) {
        metre_meas_new_tracks.reserve(
                unconfirmed_associations.unmatched_det_indices.size());
    }
    for (int detection_idx: unconfirmed_associations.unmatched_det_indices)
    {
        const std::shared_ptr<Track> &detection =
                unmatched_detections_after_1st_association[detection_idx];
        unmatched_high_conf_detections.push_back(detection);
        if (pitch_pass_active) {
            metre_meas_new_tracks.push_back(
                    metre_meas_unmatched_high_conf[detection_idx]);
        }
    }

    // Initialize new tracks for the high confidence detections left after all the associations
    for (size_t new_idx = 0;
         new_idx < unmatched_high_conf_detections.size(); ++new_idx)
    {
        const std::shared_ptr<Track> &detection =
                unmatched_high_conf_detections[new_idx];
        if (detection->get_score() >= _new_track_thresh)
        {
            detection->activate(*_kalman_filter, _frame_id);
            activated_tracks.push_back(detection);

            // Phase E.2 — seed metre state on newborn tracks when the host
            // provided a metre measurement. update_pitch lazy-activates with
            // velocity = 0, so the second frame's predict gives a sensible
            // prior even without an explicit activate_pitch call.
            if (pitch_pass_active) {
                const auto& maybe_metre = metre_meas_new_tracks[new_idx];
                if (maybe_metre.has_value()) {
                    detection->update_pitch(*_pitch_kalman_filter,
                                            maybe_metre.value());
                }
            }
        }
    }
    ////////////////// Initialize new tracks //////////////////


    ////////////////// Update lost tracks state //////////////////
    for (const std::shared_ptr<Track> &track: _lost_tracks)
    {
        if (_frame_id - track->end_frame() > _max_time_lost)
        {
            track->mark_removed();
            removed_tracks.push_back(track);
        }
    }
    ////////////////// Update lost tracks state //////////////////


    ////////////////// Clean up the track lists //////////////////
    std::vector<std::shared_ptr<Track>> updated_tracked_tracks;
    for (const std::shared_ptr<Track> &_tracked_track: _tracked_tracks)
    {
        if (_tracked_track->state == TrackState::Tracked)
        {
            updated_tracked_tracks.push_back(_tracked_track);
        }
    }
    _tracked_tracks =
            _merge_track_lists(updated_tracked_tracks, activated_tracks);
    _tracked_tracks = _merge_track_lists(_tracked_tracks, refind_tracks);

    _lost_tracks = _merge_track_lists(_lost_tracks, lost_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, _tracked_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, removed_tracks);

    std::vector<std::shared_ptr<Track>> tracked_tracks_cleaned,
            lost_tracks_cleaned;
    _remove_duplicate_tracks(tracked_tracks_cleaned, lost_tracks_cleaned,
                             _tracked_tracks, _lost_tracks);
    _tracked_tracks = tracked_tracks_cleaned,
    _lost_tracks = lost_tracks_cleaned;
    ////////////////// Clean up the track lists //////////////////


    ////////////////// Update output tracks //////////////////
    std::vector<std::shared_ptr<Track>> output_tracks;
    for (const std::shared_ptr<Track> &track: _tracked_tracks)
    {
        if (track->is_activated)
        {
            output_tracks.push_back(track);
        }
    }
    ////////////////// Update output tracks //////////////////

    return output_tracks;
}


void BoTSORT::set_gmc_enabled(bool enabled) noexcept
{
    // Guard against re-enabling GMC when no algorithm was constructed —
    // _gmc_algo->apply() would crash. Construction-time absence of GMC
    // (gmc_config empty) means the caller cannot opt back in mid-run.
    if (enabled && !_gmc_algo)
    {
        return;
    }
    _gmc_enabled = enabled;
}


FeatureVector BoTSORT::_extract_features(const cv::Mat &frame,
                                         const cv::Rect_<float> &bbox_tlwh)
{
    cv::Mat patch = frame(bbox_tlwh);
    return _reid_model->extract_features(patch);
}


std::vector<std::shared_ptr<Track>>
BoTSORT::_merge_track_lists(std::vector<std::shared_ptr<Track>> &tracks_list_a,
                            std::vector<std::shared_ptr<Track>> &tracks_list_b)
{
    std::map<int, bool> exists;
    std::vector<std::shared_ptr<Track>> merged_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_list_a)
    {
        exists[track->track_id] = true;
        merged_tracks_list.push_back(track);
    }

    for (const std::shared_ptr<Track> &track: tracks_list_b)
    {
        if (exists.find(track->track_id) == exists.end())
        {
            exists[track->track_id] = true;
            merged_tracks_list.push_back(track);
        }
    }

    return merged_tracks_list;
}


std::vector<std::shared_ptr<Track>> BoTSORT::_remove_from_list(
        std::vector<std::shared_ptr<Track>> &tracks_list,
        std::vector<std::shared_ptr<Track>> &tracks_to_remove)
{
    std::map<int, bool> exists;
    std::vector<std::shared_ptr<Track>> new_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_to_remove)
    {
        exists[track->track_id] = true;
    }

    for (const std::shared_ptr<Track> &track: tracks_list)
    {
        if (exists.find(track->track_id) == exists.end())
        {
            new_tracks_list.push_back(track);
        }
    }

    return new_tracks_list;
}


void BoTSORT::_remove_duplicate_tracks(
        std::vector<std::shared_ptr<Track>> &result_tracks_a,
        std::vector<std::shared_ptr<Track>> &result_tracks_b,
        std::vector<std::shared_ptr<Track>> &tracks_list_a,
        std::vector<std::shared_ptr<Track>> &tracks_list_b)
{
    CostMatrix iou_dists = iou_distance(tracks_list_a, tracks_list_b);

    std::unordered_set<size_t> dup_a, dup_b;
    for (Eigen::Index i = 0; i < iou_dists.rows(); i++)
    {
        for (Eigen::Index j = 0; j < iou_dists.cols(); j++)
        {
            if (iou_dists(i, j) < 0.15)
            {
                int time_a = static_cast<int>(tracks_list_a[i]->frame_id -
                                              tracks_list_a[i]->start_frame);
                int time_b = static_cast<int>(tracks_list_b[j]->frame_id -
                                              tracks_list_b[j]->start_frame);

                // We make an assumption that the longer trajectory is the correct one
                if (time_a > time_b)
                {
                    dup_b.insert(
                            j);// In list b, track with index j is a duplicate
                }
                else
                {
                    dup_a.insert(
                            i);// In list a, track with index i is a duplicate
                }
            }
        }
    }

    // Remove duplicates from the lists
    for (size_t i = 0; i < tracks_list_a.size(); i++)
    {
        if (dup_a.find(i) == dup_a.end())
        {
            result_tracks_a.push_back(tracks_list_a[i]);
        }
    }

    for (size_t i = 0; i < tracks_list_b.size(); i++)
    {
        if (dup_b.find(i) == dup_b.end())
        {
            result_tracks_b.push_back(tracks_list_b[i]);
        }
    }
}

void BoTSORT::_load_params_from_config(const TrackerParams &config)
{
    _reid_enabled = config.reid_enabled;
    _gmc_enabled = config.gmc_enabled;
    _track_high_thresh = config.track_high_thresh;
    _track_low_thresh = config.track_low_thresh;
    _new_track_thresh = config.new_track_thresh;
    _track_buffer = config.track_buffer;
    _match_thresh = config.match_thresh;
    _proximity_thresh = config.proximity_thresh;
    _appearance_thresh = config.appearance_thresh;
    _gmc_method_name = config.gmc_method_name;
    _frame_rate = config.frame_rate;
    _lambda = config.lambda;
}
#include "GlobalMotionCompensation.h"

std::map<const char *, GMC_Method> GMC_method_map = {
        {"orb", GMC_Method::ORB},
        {"ecc", GMC_Method::ECC},
        {"sparseOptFlow", GMC_Method::SparseOptFlow},
        {"optFlowModified", GMC_Method::OptFlowModified},
};


GlobalMotionCompensation::GlobalMotionCompensation(GMC_Method method, float downscale) {
    if (method == GMC_Method::ORB) {
        _gmc_algorithm = std::make_unique<ORB_GMC>(downscale);
    } else if (method == GMC_Method::ECC) {
        _gmc_algorithm = std::make_unique<ECC_GMC>(downscale);
    } else if (method == GMC_Method::SparseOptFlow) {
        _gmc_algorithm = std::make_unique<SparseOptFlow_GMC>(downscale);
    } else if (method == GMC_Method::OptFlowModified) {
        _gmc_algorithm = std::make_unique<OptFlowModified_GMC>(downscale);
    } else {
        throw std::runtime_error("Unknown global motion compensation method: " + method);
    }
}

HomographyMatrix GlobalMotionCompensation::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return _gmc_algorithm->apply(frame, detections);
}


// ORB
ORB_GMC::ORB_GMC(float downscale) : _downscale(downscale) {
    _detector = cv::FastFeatureDetector::create();
    _extractor = cv::ORB::create();
    _matcher = cv::BFMatcher::create(cv::NORM_HAMMING);// Brute Force Matcher
}

HomographyMatrix ORB_GMC::apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) {
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;

    HomographyMatrix H;
    H.setIdentity();

    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);


    // Downscale
    if (_downscale > 1.0F) {
        width /= _downscale, height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }

    // Create a mask, corner regions are ignored
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
    cv::Rect roi(width * 0.02, height * 0.02, width * 0.96, height * 0.96);
    mask(roi) = 255;


    // Set all the foreground (area with detections) to 0
    // This is to prevent the algorithm from detecting keypoints in the foreground so CMC can work better
    for (const auto &det: detections) {
        cv::Rect tlwh_downscaled(det.bbox_tlwh.x / _downscale, det.bbox_tlwh.y / _downscale,
                                 det.bbox_tlwh.width / _downscale, det.bbox_tlwh.height / _downscale);
        mask(tlwh_downscaled) = 0;
    }


    // Detect keypoints in background
    std::vector<cv::KeyPoint> keypoints;
    _detector->detect(frame, keypoints, mask);


    // Extract descriptors for the detected keypoints
    cv::Mat descriptors;
    _extractor->compute(frame, keypoints, descriptors);

    if (!_first_frame_initialized) {
        /**
         *  If this is the first frame, there is nothing to match
         *  Save the keypoints and descriptors, return identity matrix 
         */
        _first_frame_initialized = true;
        _prev_frame = frame.clone();
        _prev_keypoints = keypoints;
        _prev_descriptors = descriptors.clone();
        return H;
    }


    // Match descriptors between the current frame and the previous frame
    std::vector<std::vector<cv::DMatch>> knn_matches;
    _matcher->knnMatch(_prev_descriptors, descriptors, knn_matches, 2);


    // Filter matches on the basis of spatial distance
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> spatial_distances;
    cv::Point2f max_spatial_distance(0.25 * width, 0.25 * height);

    for (const auto &knnMatch: knn_matches) {
        const auto &m = knnMatch[0];
        const auto &n = knnMatch[1];

        // Check the distance between the previous and current match for the same keypoint
        if (m.distance < 0.9 * n.distance) {
            cv::Point2f prev_keypoint_location = _prev_keypoints[m.queryIdx].pt;
            cv::Point2f curr_keypoint_location = keypoints[m.trainIdx].pt;

            cv::Point2f distance = prev_keypoint_location - curr_keypoint_location;

            if (cv::abs(distance.x) < max_spatial_distance.x && cv::abs(distance.y) < max_spatial_distance.y) {
                spatial_distances.push_back(distance);
                matches.push_back(m);
            }
        }
    }


    // If couldn't find any matches, return identity matrix
    if (matches.empty()) {
        _prev_frame = frame.clone();
        _prev_keypoints = keypoints;
        _prev_descriptors = descriptors.clone();
        return H;
    }


    // Calculate mean and standard deviation of spatial distances
    cv::Scalar mean_spatial_distance, std_spatial_distance;
    cv::meanStdDev(spatial_distances, mean_spatial_distance, std_spatial_distance);

    // Get good matches, i.e. points that are within 2.5 standard deviations of the mean spatial distance
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> prev_points, curr_points;
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::Point2f mean_normalized_sd(spatial_distances[i].x - mean_spatial_distance[0], spatial_distances[i].y - mean_spatial_distance[1]);
        if (mean_normalized_sd.x < 2.5 * std_spatial_distance[0] && mean_normalized_sd.y < 2.5 * std_spatial_distance[1]) {
            prev_points.push_back(_prev_keypoints[matches[i].queryIdx].pt);
            curr_points.push_back(keypoints[matches[i].trainIdx].pt);
        }
    }


    // Find the rigid transformation between the previous and current frame on the basis of the good matches
    if (prev_points.size() > 4) {
        cv::Mat inliers;
        cv::Mat affine_matrix = cv::estimateAffinePartial2D(prev_points, curr_points, inliers, cv::RANSAC);

        double inlier_ratio = cv::countNonZero(inliers) / (double) inliers.rows;
        if (inlier_ratio > _inlier_ratio) {
            cv2eigen(affine_matrix, H);
            if (_downscale > 1.0) {
                H(0, 2) *= _downscale;
                H(1, 2) *= _downscale;
            }
        } else {
            std::cout << "Warning: Could not estimate affine matrix" << std::endl;
        }
    }

#ifdef DEBUG
    cv::Mat matches_img;
    cv::hconcat(_prev_frame, frame, matches_img);
    cv::cvtColor(matches_img, matches_img, cv::COLOR_GRAY2BGR);

    int W = _prev_frame.cols;

    for (const auto &m: good_matches) {
        cv::Point prev_pt = _prev_keypoints[m.queryIdx].pt;
        cv::Point curr_pt = keypoints[m.trainIdx].pt;

        curr_pt.x += W;

        cv::Scalar color = cv::Scalar::all(rand() % 255);
        color = cv::Scalar((int) color[0], (int) color[1], (int) color[2]);

        cv::line(matches_img, prev_pt, curr_pt, color, 1, cv::LineTypes::LINE_AA);
        cv::circle(matches_img, prev_pt, 2, color, -1);
        cv::circle(matches_img, curr_pt, 2, color, -1);
    }

    cv::imshow("Matches", matches_img);

#endif


    // Update previous frame, keypoints and descriptors
    _prev_frame = frame.clone();
    _prev_keypoints = keypoints;
    _prev_descriptors = descriptors.clone();
    return H;
}


// ECC
ECC_GMC::ECC_GMC(float downscale, int max_iterations, int termination_eps)
    : _downscale(downscale) {
    _termination_criteria = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, max_iterations, termination_eps);
}

HomographyMatrix ECC_GMC::apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) {
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;

    HomographyMatrix H;
    H.setIdentity();

    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);


    // Downscale
    if (_downscale > 1.0F) {
        width /= _downscale, height /= _downscale;
        cv::GaussianBlur(frame, frame, _gaussian_blur_kernel_size, 1.5);
        cv::resize(frame, frame, cv::Size(width, height));
    }

    if (!_first_frame_initialized) {
        /**
         *  If this is the first frame, there is nothing to match
         *  Save the keypoints and descriptors, return identity matrix
         */
        _first_frame_initialized = true;
        _prev_frame = frame.clone();
        return H;
    }

    try {
        cv::Mat H_cvMat;
        cv::findTransformECC(_prev_frame, frame, H_cvMat, cv::MOTION_EUCLIDEAN, _termination_criteria, cv::noArray(), 1);
        cv2eigen(H_cvMat, H);
        _prev_frame = frame.clone();
    } catch (const cv::Exception &e) {
        std::cout << "Warning: Could not estimate affine matrix" << std::endl;
    }


    return H;
}


// Optical Flow
SparseOptFlow_GMC::SparseOptFlow_GMC(float downscale) : _downscale(downscale) {}
HomographyMatrix SparseOptFlow_GMC::apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) {
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;

    HomographyMatrix H;
    H.setIdentity();

    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);


    // Downscale
    if (_downscale > 1.0F) {
        width /= _downscale, height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }


    // Detect keypoints
    std::vector<cv::Point2f> keypoints;
    cv::goodFeaturesToTrack(frame, keypoints, _maxCorners, _qualityLevel, _minDistance, cv::noArray(), _blockSize, _useHarrisDetector, _k);

    if (!_first_frame_initialized) {
        /**
         *  If this is the first frame, there is nothing to match
         *  Save the keypoints and descriptors, return identity matrix 
         */
        _first_frame_initialized = true;
        _prev_frame = frame.clone();
        _prev_keypoints = keypoints;
        return H;
    }


    // Find correspondences between the previous and current frame
    std::vector<cv::Point2f> matched_keypoints;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(_prev_frame, frame, _prev_keypoints, matched_keypoints, status, err);


    // Keep good matches
    std::vector<cv::Point2f> prev_points, curr_points;
    for (size_t i = 0; i < matched_keypoints.size(); i++) {
        if (status[i]) {
            prev_points.push_back(_prev_keypoints[i]);
            curr_points.push_back(matched_keypoints[i]);
        }
    }


    // Estimate affine matrix
    if (prev_points.size() > 4) {
        cv::Mat inliers;
        cv::Mat affine_matrix = cv::estimateAffinePartial2D(prev_points, curr_points, inliers, cv::RANSAC);

        double inlier_ratio = cv::countNonZero(inliers) / (double) inliers.rows;
        if (inlier_ratio > _inlier_ratio) {
            cv2eigen(affine_matrix, H);
            if (_downscale > 1.0) {
                H(0, 2) *= _downscale;
                H(1, 2) *= _downscale;
            }
        } else {
            std::cout << "Warning: Could not estimate affine matrix" << std::endl;
        }
    }

    _prev_frame = frame.clone();
    _prev_keypoints = keypoints;
    return H;
}


// Optical Flow Modified
OptFlowModified_GMC::OptFlowModified_GMC(float downscale) : _downscale(downscale) {}
HomographyMatrix OptFlowModified_GMC::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    HomographyMatrix H;
    H.setIdentity();

    std::cout << "Warning: OptFlowModified_GMC not implemented, returning identity matrix" << std::endl;
    return H;
}

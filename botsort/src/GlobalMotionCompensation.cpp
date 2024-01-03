#include "GlobalMotionCompensation.h"

#include <opencv2/videostab/global_motion.hpp>
#include <opencv2/videostab/motion_core.hpp>

#include "INIReader.h"

std::map<std::string, GMC_Method> GlobalMotionCompensation::GMC_method_map = {
        {"orb", GMC_Method::ORB},
        {"ecc", GMC_Method::ECC},
        {"sparseOptFlow", GMC_Method::SparseOptFlow},
        {"optFlowModified", GMC_Method::OptFlowModified},
        {"OpenCV_VideoStab", GMC_Method::OpenCV_VideoStab},
};


GlobalMotionCompensation::GlobalMotionCompensation(
        GMC_Method method, const std::string &config_path)
{
    if (method == GMC_Method::ORB)
    {
        std::cout << "Using ORB for GMC" << std::endl;
        _gmc_algorithm = std::make_unique<ORB_GMC>(config_path);
    }
    else if (method == GMC_Method::ECC)
    {
        std::cout << "Using ECC for GMC" << std::endl;
        _gmc_algorithm = std::make_unique<ECC_GMC>(config_path);
    }
    else if (method == GMC_Method::SparseOptFlow)
    {
        std::cout << "Using SparseOptFlow for GMC" << std::endl;
        _gmc_algorithm = std::make_unique<SparseOptFlow_GMC>(config_path);
    }
    else if (method == GMC_Method::OptFlowModified)
    {
        std::cout << "Using OptFlowModified for GMC" << std::endl;
        _gmc_algorithm = std::make_unique<OptFlowModified_GMC>(config_path);
    }
    else if (method == GMC_Method::OpenCV_VideoStab)
    {
        std::cout << "Using OpenCV_VideoStab for GMC" << std::endl;
        _gmc_algorithm = std::make_unique<OpenCV_VideoStab_GMC>(config_path);
    }
    else
    {
        throw std::runtime_error("Unknown global motion compensation method: " +
                                 std::to_string(method));
    }
}


HomographyMatrix
GlobalMotionCompensation::apply(const cv::Mat &frame,
                                const std::vector<Detection> &detections)
{
    return _gmc_algorithm->apply(frame, detections);
}


// ORB
ORB_GMC::ORB_GMC(const std::string &config_path)
{
    _load_params_from_config(config_path);

    _detector = cv::FastFeatureDetector::create();
    _extractor = cv::ORB::create();
    _matcher = cv::BFMatcher::create(cv::NORM_HAMMING);// Brute Force Matcher
}


void ORB_GMC::_load_params_from_config(const std::string &config_path)
{
    INIReader gmc_config(config_path);
    if (gmc_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    _downscale = gmc_config.GetFloat(_algo_name, "downscale", 2.0);
    _inlier_ratio = gmc_config.GetFloat(_algo_name, "inlier_ratio", 0.5);
    _ransac_conf = gmc_config.GetFloat(_algo_name, "ransac_conf", 0.99);
    _ransac_max_iters =
            gmc_config.GetInteger(_algo_name, "ransac_max_iters", 500);
}


HomographyMatrix ORB_GMC::apply(const cv::Mat &frame_raw,
                                const std::vector<Detection> &detections)
{
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;

    HomographyMatrix H;
    H.setIdentity();

    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);


    // Downscale
    if (_downscale > 1.0F)
    {
        width /= _downscale, height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }

    // Create a mask, corner regions are ignored
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
    cv::Rect roi(
            static_cast<int>(width * 0.02), static_cast<int>(height * 0.02),
            static_cast<int>(width * 0.96), static_cast<int>(height * 0.96));
    mask(roi) = 255;


    // Set all the foreground (area with detections) to 0
    // This is to prevent the algorithm from detecting keypoints in the foreground so CMC can work better
    for (const auto &det: detections)
    {
        cv::Rect tlwh_downscaled(
                static_cast<int>(det.bbox_tlwh.x / _downscale),
                static_cast<int>(det.bbox_tlwh.y / _downscale),
                static_cast<int>(det.bbox_tlwh.width / _downscale),
                static_cast<int>(det.bbox_tlwh.height / _downscale));
        mask(tlwh_downscaled) = 0;
    }


    // Detect keypoints in background
    std::vector<cv::KeyPoint> keypoints;
    _detector->detect(frame, keypoints, mask);


    // Extract descriptors for the detected keypoints
    cv::Mat descriptors;
    _extractor->compute(frame, keypoints, descriptors);

    if (!_first_frame_initialized)
    {
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
    cv::Point2f max_spatial_distance(0.25F * width, 0.25F * height);

    for (const auto &knnMatch: knn_matches)
    {
        const auto &m = knnMatch[0];
        const auto &n = knnMatch[1];

        // Check the distance between the previous and current match for the same keypoint
        if (m.distance < 0.9 * n.distance)
        {
            cv::Point2f prev_keypoint_location = _prev_keypoints[m.queryIdx].pt;
            cv::Point2f curr_keypoint_location = keypoints[m.trainIdx].pt;

            cv::Point2f distance =
                    prev_keypoint_location - curr_keypoint_location;

            if (cv::abs(distance.x) < max_spatial_distance.x &&
                cv::abs(distance.y) < max_spatial_distance.y)
            {
                spatial_distances.push_back(distance);
                matches.push_back(m);
            }
        }
    }


    // If couldn't find any matches, return identity matrix
    if (matches.empty())
    {
        _prev_frame = frame.clone();
        _prev_keypoints = keypoints;
        _prev_descriptors = descriptors.clone();
        return H;
    }


    // Calculate mean and standard deviation of spatial distances
    cv::Scalar mean_spatial_distance, std_spatial_distance;
    cv::meanStdDev(spatial_distances, mean_spatial_distance,
                   std_spatial_distance);

    // Get good matches, i.e. points that are within 2.5 standard deviations of the mean spatial distance
    std::vector<cv::DMatch> good_matches;
    std::vector<cv::Point2f> prev_points, curr_points;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        cv::Point2f mean_normalized_sd(
                spatial_distances[i].x - mean_spatial_distance[0],
                spatial_distances[i].y - mean_spatial_distance[1]);
        if (mean_normalized_sd.x < 2.5 * std_spatial_distance[0] &&
            mean_normalized_sd.y < 2.5 * std_spatial_distance[1])
        {
            prev_points.push_back(_prev_keypoints[matches[i].queryIdx].pt);
            curr_points.push_back(keypoints[matches[i].trainIdx].pt);
        }
    }


    // Find the rigid transformation between the previous and current frame on the basis of the good matches
    if (prev_points.size() > 4)
    {
        cv::Mat inliers;
        cv::Mat homography =
                cv::findHomography(prev_points, curr_points, cv::RANSAC, 3,
                                   inliers, _ransac_max_iters, _ransac_conf);

        double inlier_ratio = cv::countNonZero(inliers) / (double) inliers.rows;
        if (inlier_ratio > _inlier_ratio)
        {
            cv2eigen(homography, H);
            if (_downscale > 1.0)
            {
                H(0, 2) *= _downscale;
                H(1, 2) *= _downscale;
            }
        }
        else
        {
            std::cout << "Warning: Could not estimate affine matrix"
                      << std::endl;
        }
    }

#ifdef DEBUG
    cv::Mat matches_img;
    cv::hconcat(_prev_frame, frame, matches_img);
    cv::cvtColor(matches_img, matches_img, cv::COLOR_GRAY2BGR);

    int W = _prev_frame.cols;

    for (const auto &m: good_matches)
    {
        cv::Point prev_pt = _prev_keypoints[m.queryIdx].pt;
        cv::Point curr_pt = keypoints[m.trainIdx].pt;

        curr_pt.x += W;

        cv::Scalar color = cv::Scalar::all(rand() % 255);
        color = cv::Scalar((int) color[0], (int) color[1], (int) color[2]);

        cv::line(matches_img, prev_pt, curr_pt, color, 1,
                 cv::LineTypes::LINE_AA);
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
ECC_GMC::ECC_GMC(const std::string &config_path)
{
    _load_params_from_config(config_path);

    _termination_criteria =
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                             _max_iterations, _termination_eps);
}


void ECC_GMC::_load_params_from_config(const std::string &config_path)
{
    INIReader gmc_config(config_path);
    if (gmc_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    _downscale = gmc_config.GetFloat(_algo_name, "downscale", 5.0F);
    _max_iterations = gmc_config.GetInteger(_algo_name, "max_iterations", 100);
    _termination_eps = gmc_config.GetFloat(_algo_name, "termination_eps", 1e-6);
}


HomographyMatrix ECC_GMC::apply(const cv::Mat &frame_raw,
                                const std::vector<Detection> &detections)
{
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;

    HomographyMatrix H;
    H.setIdentity();

    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);


    // Downscale
    if (_downscale > 1.0F)
    {
        width /= _downscale, height /= _downscale;
        cv::GaussianBlur(frame, frame, _gaussian_blur_kernel_size, 1.5);
        cv::resize(frame, frame, cv::Size(width, height));
    }

    if (!_first_frame_initialized)
    {
        /**
         *  If this is the first frame, there is nothing to match
         *  Save the keypoints and descriptors, return identity matrix
         */
        _first_frame_initialized = true;
        _prev_frame = frame.clone();
        return H;
    }

    try
    {
        cv::Mat H_cvMat;
#if CV_MAJOR_VERSION == 3
        cv::findTransformECC(_prev_frame, frame, H_cvMat, cv::MOTION_EUCLIDEAN,
                             _termination_criteria);
#elif CV_MAJOR_VERSION == 4
        cv::findTransformECC(_prev_frame, frame, H_cvMat, cv::MOTION_EUCLIDEAN,
                             _termination_criteria, cv::noArray(), 1);
#endif
        cv2eigen(H_cvMat, H);
        _prev_frame = frame.clone();
    }
    catch (const cv::Exception &e)
    {
        std::cout << "Warning: Could not estimate affine matrix" << std::endl;
    }


    return H;
}


// Optical Flow
SparseOptFlow_GMC::SparseOptFlow_GMC(const std::string &config_path)
{
    _load_params_from_config(config_path);
}


void SparseOptFlow_GMC::_load_params_from_config(const std::string &config_path)
{
    INIReader gmc_config(config_path);
    if (gmc_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    _useHarrisDetector =
            gmc_config.GetBoolean(_algo_name, "use_harris_detector", false);

    _maxCorners = gmc_config.GetInteger(_algo_name, "max_corners", 1000);
    _blockSize = gmc_config.GetInteger(_algo_name, "block_size", 3);
    _ransac_max_iters =
            gmc_config.GetInteger(_algo_name, "ransac_max_iters", 500);

    _qualityLevel = gmc_config.GetReal(_algo_name, "quality_level", 0.01);
    _k = gmc_config.GetReal(_algo_name, "k", 0.04);
    _minDistance = gmc_config.GetReal(_algo_name, "min_distance", 1.0);


    _downscale = gmc_config.GetFloat(_algo_name, "downscale", 2.0F);
    _inlier_ratio = gmc_config.GetFloat(_algo_name, "inlier_ratio", 0.5);
    _ransac_conf = gmc_config.GetFloat(_algo_name, "ransac_conf", 0.99);
}


HomographyMatrix
SparseOptFlow_GMC::apply(const cv::Mat &frame_raw,
                         const std::vector<Detection> &detections)
{
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;

    HomographyMatrix H;
    H.setIdentity();

    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);


    // Downscale
    if (_downscale > 1.0F)
    {
        width /= _downscale, height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }


    // Detect keypoints
    std::vector<cv::Point2f> keypoints;
    cv::goodFeaturesToTrack(frame, keypoints, _maxCorners, _qualityLevel,
                            _minDistance, cv::noArray(), _blockSize,
                            _useHarrisDetector, _k);

    if (!_first_frame_initialized || _prev_keypoints.size() == 0)
    {
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
    try
    {
        cv::calcOpticalFlowPyrLK(_prev_frame, frame, _prev_keypoints,
                                 matched_keypoints, status, err);
    }
    catch (const cv::Exception &e)
    {
        std::cout << "Warning: Could not find correspondences for GMC"
                  << std::endl;
        return H;
    }


    // Keep good matches
    std::vector<cv::Point2f> prev_points, curr_points;
    for (size_t i = 0; i < matched_keypoints.size(); i++)
    {
        if (status[i])
        {
            prev_points.push_back(_prev_keypoints[i]);
            curr_points.push_back(matched_keypoints[i]);
        }
    }


    // Estimate affine matrix
    if (prev_points.size() > 4)
    {
        cv::Mat inliers;
        cv::Mat homography =
                cv::findHomography(prev_points, curr_points, cv::RANSAC, 3,
                                   inliers, _ransac_max_iters, _ransac_conf);

        double inlier_ratio = cv::countNonZero(inliers) / (double) inliers.rows;
        if (inlier_ratio > _inlier_ratio)
        {
            cv2eigen(homography, H);
            if (_downscale > 1.0)
            {
                H(0, 2) *= _downscale;
                H(1, 2) *= _downscale;
            }
        }
        else
        {
            std::cout << "Warning: Could not estimate affine matrix"
                      << std::endl;
        }
    }

    _prev_frame = frame.clone();
    _prev_keypoints = keypoints;
    return H;
}


// OpenCV VideoStab
OpenCV_VideoStab_GMC::OpenCV_VideoStab_GMC(const std::string &config_path)
{
    _load_params_from_config(config_path);

    _motion_estimator = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(
            cv::videostab::MM_SIMILARITY);

    _keypoint_motion_estimator =
            cv::makePtr<cv::videostab::KeypointBasedMotionEstimator>(
                    _motion_estimator);
    _keypoint_motion_estimator->setDetector(
            cv::GFTTDetector::create(_num_features));
}


void OpenCV_VideoStab_GMC::_load_params_from_config(
        const std::string &config_path)
{
    INIReader gmc_config(config_path);
    if (gmc_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    _downscale = gmc_config.GetFloat(_algo_name, "downscale", 2.0F);
    _num_features = gmc_config.GetInteger(_algo_name, "num_features", 4000);
    _detections_masking =
            gmc_config.GetBoolean(_algo_name, "detections_masking", true);
}


HomographyMatrix
OpenCV_VideoStab_GMC::apply(const cv::Mat &frame_raw,
                            const std::vector<Detection> &detections)
{
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;

    HomographyMatrix H;
    H.setIdentity();
    cv::Mat frame = frame_raw.clone();

    if (frame_raw.empty())
    {
        return H;
    }

    // Downscale
    if (_downscale > 1.0F)
    {
        width /= _downscale, height /= _downscale;
        cv::resize(frame_raw, frame, cv::Size(width, height));
    }

    cv::Mat homography = cv::Mat::eye(3, 3, CV_32F);

    if (!_prev_frame.empty())
    {
        if (_detections_masking)
        {
            cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
            for (const Detection &detection: detections)
            {
                cv::Rect rect = detection.bbox_tlwh;
                rect.x /= _downscale;
                rect.y /= _downscale;
                rect.width /= _downscale;
                rect.height /= _downscale;
                mask(rect) = 255;
            }

            _keypoint_motion_estimator->setFrameMask(mask);
        }

        bool ok;
        homography =
                _keypoint_motion_estimator->estimate(_prev_frame, frame, &ok);

        if (ok)
        {
            cv2eigen(homography, H);
            if (_downscale > 1.0)
            {
                H(0, 2) *= _downscale;
                H(1, 2) *= _downscale;
            }
        }
    }

    frame.copyTo(_prev_frame);
    homography.copyTo(_prev_homography);
    return H;
}


// Optical Flow Modified
OptFlowModified_GMC::OptFlowModified_GMC(const std::string &config_path)
{
    _load_params_from_config(config_path);
}


void OptFlowModified_GMC::_load_params_from_config(
        const std::string &config_path)
{
    INIReader gmc_config(config_path);
    if (gmc_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    _downscale = gmc_config.GetFloat(_algo_name, "downscale", 2.0F);
}


HomographyMatrix
OptFlowModified_GMC::apply(const cv::Mat &frame,
                           const std::vector<Detection> &detections)
{
    HomographyMatrix H;
    H.setIdentity();

    std::cout << "Warning: OptFlowModified_GMC not implemented, returning "
                 "identity matrix"
              << std::endl;
    return H;
}

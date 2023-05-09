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
        auto _gmc_algorithm = std::make_unique<ECC_GMC>(downscale);
    } else if (method == GMC_Method::SparseOptFlow) {
        auto _gmc_algorithm = std::make_unique<SparseOptFlow_GMC>(downscale);
    } else if (method == GMC_Method::OptFlowModified) {
        auto _gmc_algorithm = std::make_unique<OptFlowModified_GMC>(downscale);
    } else {
        throw std::runtime_error("Unknown global motion compensation method: " + method);
    }
}

HomographyMatrix GlobalMotionCompensation::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return _gmc_algorithm->apply(frame, detections);
}


// ORB
ORB_GMC::ORB_GMC(float downscale) : _downscale(downscale) {
    _detector = cv::FastFeatureDetector::create(20);
    _extractor = cv::ORB::create();
    _matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
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
    if (_downscale > 1) {
        width /= _downscale, height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }

    // Create a mask, corner regions are ignored
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
    cv::Rect roi(width * 0.02, height * 0.02, width * 0.096, height * 0.96);
    mask(roi) = 255;


    // Set all the foreground (areas with detections) to 0
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
         *  Just save the keypoints and descriptors, return identity matrix 
         */
        _first_frame_initialized = true;
        _prev_frame = frame;
        _prev_keypoints = keypoints;
        _prev_descriptors = descriptors;
        return H;
    }

    // Match descriptors between the current frame and the previous frame
    std::vector<std::vector<cv::DMatch>> knn_matches;
    _matcher->knnMatch(descriptors, _prev_descriptors, knn_matches, 2);

    // Filter matches on the basis of smallest spatial distance
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point2f> spatial_distances;
    cv::Point2f max_spatial_distance(0.25 * width, 0.25 * height);


    // TODO: Complete this
    return HomographyMatrix();
}


// ECC
ECC_GMC::ECC_GMC(float downscale) : _downscale(downscale) {}
HomographyMatrix ECC_GMC::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return HomographyMatrix();
}


// Optical Flow
SparseOptFlow_GMC::SparseOptFlow_GMC(float downscale) : _downscale(downscale) {}
HomographyMatrix SparseOptFlow_GMC::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return HomographyMatrix();
}


// Optical Flow Modified
OptFlowModified_GMC::OptFlowModified_GMC(float downscale) : _downscale(downscale) {}
HomographyMatrix OptFlowModified_GMC::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return HomographyMatrix();
}

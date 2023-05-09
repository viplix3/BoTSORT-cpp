#include "GlobalMotionCompensation.h"

std::map<const char *, GMC_Method> GMC_method_map = {
        {"orb", GMC_Method::ORB},
        {"sift", GMC_Method::SIFT},
        {"ecc", GMC_Method::ECC},
        {"sparseOptFlow", GMC_Method::SparseOptFlow}};


GlobalMotionCompensation::GlobalMotionCompensation(GMC_Method method, uint8_t downscale) {
    if (method == GMC_Method::ORB) {
        _gmc_algorithm = std::make_unique<ORB_GMC>(downscale);
    } else if (method == GMC_Method::SIFT) {
        auto _gmc_algorithm = std::make_unique<SIFT_GMC>(downscale);
    } else if (method == GMC_Method::ECC) {
        auto _gmc_algorithm = std::make_unique<ECC_GMC>(downscale);
    } else if (method == GMC_Method::SparseOptFlow) {
        auto _gmc_algorithm = std::make_unique<SparseOptFlow_GMC>(downscale);
    } else {
        throw std::runtime_error("Unknown global motion compensation method: " + method);
    }
}

HomographyMatrix GlobalMotionCompensation::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return _gmc_algorithm->apply(frame, detections);
}


// ORB
ORB_GMC::ORB_GMC(uint8_t downscale) : _downscale(downscale) {}

HomographyMatrix ORB_GMC::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    // Initialization
    int height = frame.rows;
    int width = frame.cols;
    HomographyMatrix H = HomographyMatrix::Identity(3, 3);

    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    // Downscale
    if (_downscale > 1) {
        width /= _downscale;
        height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }

    // TODO: Complete this
    return HomographyMatrix();
}


// SIFT
SIFT_GMC::SIFT_GMC(uint8_t downscale) : _downscale(downscale) {}
HomographyMatrix SIFT_GMC::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return HomographyMatrix();
}


// ECC
ECC_GMC::ECC_GMC(uint8_t downscale) : _downscale(downscale) {}
HomographyMatrix ECC_GMC::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return HomographyMatrix();
}


// Optical Flow
SparseOptFlow_GMC::SparseOptFlow_GMC(uint8_t downscale) : _downscale(downscale) {}
HomographyMatrix SparseOptFlow_GMC::apply(const cv::Mat &frame, const std::vector<Detection> &detections) {
    return HomographyMatrix();
}
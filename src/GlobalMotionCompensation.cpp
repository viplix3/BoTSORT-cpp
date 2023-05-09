#include "GlobalMotionCompensation.h"

std::map<const char *, GMC_Method> GMC_method_map = {
        {"orb", GMC_Method::ORB},
        {"sift", GMC_Method::SIFT},
        {"ecc", GMC_Method::ECC},
        {"sparseOptFlow", GMC_Method::SparseOptFlow}};

GlobalMotionCompensation::GlobalMotionCompensation(GMC_Method method, int downscale) {
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

void GlobalMotionCompensation::apply(const cv::Mat &frame, std::vector<float> &detections) {
    _gmc_algorithm->apply(frame, detections);
}

ORB_GMC::ORB_GMC(int downscale) : _downscale(downscale) {}

void ORB_GMC::apply(const cv::Mat &frame, std::vector<float> &detections) {
    // Initialization
    int height = frame.rows;
    int width = frame.cols;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    Eigen::MatrixXf H = Eigen::MatrixXf::Identity(2, 3);

    // Downscale
    if (_downscale > 1) {
        width /= _downscale;
        height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }

    // TODO: Complete this
}


SIFT_GMC::SIFT_GMC(int downscale) : _downscale(downscale) {}
void SIFT_GMC::apply(const cv::Mat &frame, std::vector<float> &detections) {}


ECC_GMC::ECC_GMC(int downscale) : _downscale(downscale) {}
void ECC_GMC::apply(const cv::Mat &frame, std::vector<float> &detections) {}


SparseOptFlow_GMC::SparseOptFlow_GMC(int downscale) : _downscale(downscale) {}
void SparseOptFlow_GMC::apply(const cv::Mat &frame, std::vector<float> &detections) {}
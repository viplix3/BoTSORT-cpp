#include "GlobalMotionCompensation.h"

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

GlobalMotionCompensation::~GlobalMotionCompensation() {}

void GlobalMotionCompensation::apply(const cv::Mat &frame, std::vector<float> &detections) {
    _gmc_algorithm->apply(frame, detections);
}

ORB_GMC::ORB_GMC(int downscale) : _downscale(downscale) {}

SIFT_GMC::SIFT_GMC(int downscale) : _downscale(downscale) {}

ECC_GMC::ECC_GMC(int downscale) : _downscale(downscale) {}

SparseOptFlow_GMC::SparseOptFlow_GMC(int downscale) : _downscale(downscale) {}
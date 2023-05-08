#pragma once

#include "DataType.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>

enum GMC_Method {
    ORB = 0,
    SIFT,
    ECC,
    SparseOptFlow
};

class GlobalMotionCompensation {
public:
    GlobalMotionCompensation(GMC_Method method, int downscale = 2);
    ~GlobalMotionCompensation();

    void apply(const cv::Mat &frame, std::vector<float> &detections);

private:
    std::unique_ptr<GMC_Algorithm> _gmc_algorithm;
};

class GMC_Algorithm {
public:
    virtual ~GMC_Algorithm() = default;
    virtual void apply(const cv::Mat &frame, std::vector<float> &detections) = 0;

private:
    int _downscale;
};

class ORB_GMC : public GMC_Algorithm {
public:
    ORB_GMC(int downscale = 2);
    void apply(const cv::Mat &frame, std::vector<float> &detections) override;
};

class SIFT_GMC : public GMC_Algorithm {
public:
    SIFT_GMC(int downscale = 2);
    virtual void apply(const cv::Mat &frame, std::vector<float> &detections) override;
};

class ECC_GMC : public GMC_Algorithm {
public:
    ECC_GMC(int downscale = 2);
    virtual void apply(const cv::Mat &frame, std::vector<float> &detections) override;
};

class SparseOptFlow_GMC : public GMC_Algorithm {
public:
    SparseOptFlow_GMC(int downscale = 2);
    virtual void apply(const cv::Mat &frame, std::vector<float> &detections) override;
};
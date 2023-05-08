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


class GMC_Algorithm {
public:
    virtual ~GMC_Algorithm() = default;
    virtual void apply(const cv::Mat &frame, std::vector<float> &detections) = 0;
};

class ORB_GMC : public GMC_Algorithm {
public:
    ORB_GMC(int downscale);
    void apply(const cv::Mat &frame, std::vector<float> &detections) override;

private:
    cv::Ptr<cv::FastFeatureDetector> _detector = cv::FastFeatureDetector::create(20);
    cv::Ptr<cv::ORB> _extractor = cv::ORB::create();
    cv::BFMatcher _matcher = cv::BFMatcher(cv::NORM_HAMMING);
    int _downscale;
};

class SIFT_GMC : public GMC_Algorithm {
public:
    SIFT_GMC(int downscale = 2);
    virtual void apply(const cv::Mat &frame, std::vector<float> &detections) override;

private:
    int _downscale;
};

class ECC_GMC : public GMC_Algorithm {
public:
    ECC_GMC(int downscale = 2);
    virtual void apply(const cv::Mat &frame, std::vector<float> &detections) override;

private:
    int _downscale;
};

class SparseOptFlow_GMC : public GMC_Algorithm {
public:
    SparseOptFlow_GMC(int downscale = 2);
    virtual void apply(const cv::Mat &frame, std::vector<float> &detections) override;

private:
    int _downscale;
};


class GlobalMotionCompensation {
public:
    GlobalMotionCompensation(GMC_Method method, int downscale = 2);
    ~GlobalMotionCompensation();

    void apply(const cv::Mat &frame, std::vector<float> &detections);

private:
    std::unique_ptr<GMC_Algorithm> _gmc_algorithm;
};
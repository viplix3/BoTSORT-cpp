#pragma once

#include "DataType.h"

#include <map>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>


enum GMC_Method {
    ORB = 0,
    SIFT,
    ECC,
    SparseOptFlow
};

extern std::map<const char *, GMC_Method> GMC_method_map;


class GMC_Algorithm {
public:
    virtual ~GMC_Algorithm() = default;
    virtual HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) = 0;
};

class ORB_GMC : public GMC_Algorithm {
private:
    uint8_t _downscale;
    cv::Ptr<cv::FastFeatureDetector> _detector = cv::FastFeatureDetector::create(20);
    cv::Ptr<cv::ORB> _extractor = cv::ORB::create();
    cv::BFMatcher _matcher = cv::BFMatcher(cv::NORM_HAMMING);


public:
    ORB_GMC(uint8_t downscale);
    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) override;
};

class SIFT_GMC : public GMC_Algorithm {
private:
    uint8_t _downscale;


public:
    SIFT_GMC(uint8_t downscale = 2);
    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) override;
};

class ECC_GMC : public GMC_Algorithm {
private:
    uint8_t _downscale;


public:
    ECC_GMC(uint8_t downscale = 2);
    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) override;
};

class SparseOptFlow_GMC : public GMC_Algorithm {
private:
    uint8_t _downscale;


public:
    SparseOptFlow_GMC(uint8_t downscale = 2);
    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) override;
};


class GlobalMotionCompensation {
private:
    std::unique_ptr<GMC_Algorithm> _gmc_algorithm;


public:
    GlobalMotionCompensation(GMC_Method method, uint8_t downscale = 2);
    ~GlobalMotionCompensation() = default;

    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections);
};
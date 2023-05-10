#pragma once

#include "DataType.h"

#include <map>
#include <numeric>
#include <string>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


enum GMC_Method {
    ORB = 0,
    ECC,
    SparseOptFlow,
    OptFlowModified,
};

extern std::map<const char *, GMC_Method> GMC_method_map;


class GMC_Algorithm {
public:
    virtual ~GMC_Algorithm() = default;
    virtual HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) = 0;
};

class ORB_GMC : public GMC_Algorithm {
private:
    float _downscale;
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _extractor;
    cv::Ptr<cv::DescriptorMatcher> _matcher;

    bool _first_frame_initialized = false;
    cv::Mat _prev_frame;
    std::vector<cv::KeyPoint> _prev_keypoints;
    cv::Mat _prev_descriptors;
    float _inlier_ratio = 0.05;


public:
    ORB_GMC(float downscale);

    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) override;
};

class OptFlowModified_GMC : public GMC_Algorithm {
private:
    float _downscale;


public:
    OptFlowModified_GMC(float downscale);
    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) override;
};

class ECC_GMC : public GMC_Algorithm {
private:
    float _downscale;

    bool _first_frame_initialized = false;
    cv::Mat _prev_frame;
    cv::Size _gaussian_blur_kernel_size = cv::Size(3, 3);
    cv::TermCriteria _termination_criteria;


public:
    ECC_GMC(float downscale, int max_iterations = 5000, int termination_eps = 1e-6);
    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) override;
};

class SparseOptFlow_GMC : public GMC_Algorithm {
private:
    float _downscale;


public:
    SparseOptFlow_GMC(float downscale);
    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections) override;
};


class GlobalMotionCompensation {
private:
    std::unique_ptr<GMC_Algorithm> _gmc_algorithm;


public:
    GlobalMotionCompensation(GMC_Method method, float downscale = 2.0);
    ~GlobalMotionCompensation() = default;

    HomographyMatrix apply(const cv::Mat &frame, const std::vector<Detection> &detections);
};
#pragma once

#include "DataType.h"

#include <map>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <string>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>


enum GMC_Method {
    ORB = 0,
    ECC,
    SparseOptFlow,
    OptFlowModified,
    OpenCV_VideoStab
};


class GMC_Algorithm {
public:
    virtual ~GMC_Algorithm() = default;
    virtual HomographyMatrix apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) = 0;
};

class ORB_GMC : public GMC_Algorithm {
private:
    std::string _algo_name = "orb";
    float _downscale;
    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _extractor;
    cv::Ptr<cv::DescriptorMatcher> _matcher;

    bool _first_frame_initialized = false;
    cv::Mat _prev_frame;
    std::vector<cv::KeyPoint> _prev_keypoints;
    cv::Mat _prev_descriptors;
    float _inlier_ratio, _ransac_conf;
    int _ransac_max_iters;


private:
    void _load_params_from_config(const std::string &config_dir);

public:
    explicit ORB_GMC(const std::string &config_dir);
    HomographyMatrix apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) override;
};

class ECC_GMC : public GMC_Algorithm {
private:
    std::string _algo_name = "ecc";
    float _downscale;
    int _max_iterations, _termination_eps;

    bool _first_frame_initialized = false;
    cv::Mat _prev_frame;
    cv::Size _gaussian_blur_kernel_size = cv::Size(3, 3);
    cv::TermCriteria _termination_criteria;


private:
    void _load_params_from_config(const std::string &config_dir);

public:
    explicit ECC_GMC(const std::string &config_dir);
    HomographyMatrix apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) override;
};

class SparseOptFlow_GMC : public GMC_Algorithm {
private:
    std::string _algo_name = "sparseOptFlow";
    float _downscale;

    bool _first_frame_initialized = false;
    cv::Mat _prev_frame;
    std::vector<cv::Point2f> _prev_keypoints;

    // Parameters
    int _maxCorners, _blockSize, _ransac_max_iters;
    double _qualityLevel, _k, _minDistance;
    bool _useHarrisDetector;
    float _inlier_ratio, _ransac_conf;


private:
    void _load_params_from_config(const std::string &config_dir);

public:
    explicit SparseOptFlow_GMC(const std::string &config_dir);
    HomographyMatrix apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) override;
};

class OptFlowModified_GMC : public GMC_Algorithm {
private:
    std::string _algo_name = "OptFlowModified";
    float _downscale;


private:
    void _load_params_from_config(const std::string &config_dir);

public:
    explicit OptFlowModified_GMC(const std::string &config_dir);
    HomographyMatrix apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) override;
};

class OpenCV_VideoStab_GMC : public GMC_Algorithm {
private:
    std::string _algo_name = "OpenCV_VideoStab";
    float _downscale;
    int _num_features;
    bool _detections_masking;

    cv::Mat _prev_frame;
    cv::Mat _prev_homography;

    cv::Ptr<cv::videostab::MotionEstimatorRansacL2> _motion_estimator;
    cv::Ptr<cv::videostab::KeypointBasedMotionEstimator> _keypoint_motion_estimator;


private:
    void _load_params_from_config(const std::string &config_dir);

public:
    explicit OpenCV_VideoStab_GMC(const std::string &config_dir);
    HomographyMatrix apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections) override;
};


class GlobalMotionCompensation {
public:
    static std::map<std::string, GMC_Method> GMC_method_map;

private:
    std::unique_ptr<GMC_Algorithm> _gmc_algorithm;


public:
    /**
     * @brief Construct a new Global Motion Compensation object
     * 
     * @param method GMC_Method enum member for GMC algorithm to use
     * @param config_dir Directory containing config files for GMC algorithm
     */
    explicit GlobalMotionCompensation(GMC_Method method, const std::string &config_dir);
    ~GlobalMotionCompensation() = default;

    /**
     * @brief Apply GMC algorithm to find homography matrix given frame and detections
     * 
     * @param frame_raw Input frame
     * @param detections Detections in the frame
     * @return HomographyMatrix Predicted homography matrix
     */
    HomographyMatrix apply(const cv::Mat &frame_raw, const std::vector<Detection> &detections);
};
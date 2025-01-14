#pragma once

#include <string>
#include <variant>

enum GMC_Method
{
    ORB = 0,
    ECC,
    SparseOptFlow,
    OptFlowModified,
    OpenCV_VideoStab
};

struct ORB_Params
{
    float downscale{2.f};
    float inlier_ratio{0.5f};
    float ransac_conf{.99f};
    long ransac_max_iters{500};
};

struct ECC_Params
{
    float downscale{5.f};
    long max_iterations{100};
    float termination_eps{1e-6};
};

struct SparseOptFlow_Params
{
    long max_corners{1000};
    long block_size{3};
    long ransac_max_iters{500};
    double quality_level{0.01};
    double k{0.04};
    double min_distance{1.0};
    float downscale{2.0f};
    float inlier_ratio{0.5f};
    float ransac_conf{0.99f};
    bool use_harris_detector{false};
};

struct OptFlowModified_Params
{
    float downscale{2.0f};
};

struct OpenCV_VideoStab_GMC_Params
{
    float downscale{2.0f};
    float num_features{4000};
    bool detection_masking{true};
};

struct GMC_Params
{
    using MethodParams =
            std::variant<ORB_Params, ECC_Params, SparseOptFlow_Params,
                         OptFlowModified_Params, OpenCV_VideoStab_GMC_Params,
                         std::monostate>;

    GMC_Method method_;
    MethodParams method_params_;

    static GMC_Params load_config(GMC_Method method,
                                  const std::string &config_path);
};
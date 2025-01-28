#include "GmcParams.h"

#include <iostream>

#include "DataType.h"
#include "INIReader.h"

constexpr std::size_t NUM_METHODS{5};

using MethodLoaderFunction = GMC_Params::MethodParams (*)(INIReader &);

namespace
{
GMC_Params::MethodParams load_orb_config(INIReader &gmc_config)
{
    ORB_Params params{};
    static constexpr auto SECTION = "orb";

    gmc_config.LoadFloat(SECTION, "downscale", params.downscale);
    gmc_config.LoadFloat(SECTION, "inlier_ratio", params.inlier_ratio);
    gmc_config.LoadFloat(SECTION, "ransac_conf", params.ransac_conf);
    gmc_config.LoadInteger(SECTION, "ransac_max_iters",
                           params.ransac_max_iters);

    return {params};
}

GMC_Params::MethodParams load_ecc_config(INIReader &gmc_config)
{
    ECC_Params params{};
    static constexpr auto SECTION = "ecc";

    gmc_config.LoadFloat(SECTION, "downscale", params.downscale);
    gmc_config.LoadInteger(SECTION, "max_iterations", params.max_iterations);
    gmc_config.LoadFloat(SECTION, "termination_eps", params.termination_eps);

    return {params};
}

GMC_Params::MethodParams load_sparce_config(INIReader &gmc_config)
{
    SparseOptFlow_Params params{};
    static constexpr auto SECTION = "sparseOptFlow";

    gmc_config.LoadBoolean(SECTION, "use_harris_detector",
                           params.use_harris_detector);

    gmc_config.LoadInteger(SECTION, "max_corners", params.max_corners);
    gmc_config.LoadInteger(SECTION, "block_size", params.block_size);
    gmc_config.LoadInteger(SECTION, "ransac_max_iters",
                           params.ransac_max_iters);

    gmc_config.LoadReal(SECTION, "quality_level", params.quality_level);
    gmc_config.LoadReal(SECTION, "k", params.k);
    gmc_config.LoadReal(SECTION, "min_distance", params.min_distance);

    gmc_config.LoadFloat(SECTION, "downscale", params.downscale);
    gmc_config.LoadFloat(SECTION, "inlier_ratio", params.inlier_ratio);
    gmc_config.LoadFloat(SECTION, "ransac_conf", params.ransac_conf);

    return {params};
}

GMC_Params::MethodParams load_opt_config(INIReader &gmc_config)
{
    OptFlowModified_Params params{};
    static constexpr auto SECTION = "OptFlowModified";

    gmc_config.LoadFloat(SECTION, "downscale", params.downscale);

    return params;
}

GMC_Params::MethodParams load_videostab_config(INIReader &gmc_config)
{
    OpenCV_VideoStab_GMC_Params params{};
    static constexpr auto SECTION = "OptFlowModified";

    gmc_config.LoadFloat(SECTION, "downscale", params.downscale);
    gmc_config.LoadFloat(SECTION, "num_features", params.num_features);
    gmc_config.LoadBoolean(SECTION, "detections_masking",
                           params.detection_masking);

    return params;
}
}// namespace

GMC_Params GMC_Params::load_config(GMC_Method method,
                                   const std::string &config_path)
{
    GMC_Params config;
    config.method_ = method;

    static std::array<MethodLoaderFunction, NUM_METHODS> method_loaders = {
            load_orb_config, load_ecc_config, load_sparce_config,
            load_opt_config, load_videostab_config};

    auto method_index = static_cast<std::size_t>(method);
    if (method_index >= NUM_METHODS)
    {
        throw std::runtime_error("Unknown global motion compensation method: " +
                                 std::to_string(method));
    }

    INIReader gmc_config(config_path);
    if (gmc_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    config.method_params_ =
            method_loaders[static_cast<std::size_t>(method)](gmc_config);

    return config;
}
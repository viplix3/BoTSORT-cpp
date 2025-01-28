#include "ReIDParams.h"

#include <iostream>

#include "DataType.h"
#include "INIReader.h"


ReIDParams ReIDParams::load_config(const std::string &config_path)
{
    ReIDParams config{};

    const std::string reid_name = "ReID";

    INIReader reid_config(config_path);
    if (reid_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    reid_config.LoadInteger(reid_name, "gpu_id", config.gpu_id);
    reid_config.LoadString(reid_name, "distance_metric",
                           config.distance_metric);
    reid_config.LoadInteger(reid_name, "trt_logging_level",
                            config.trt_logging_level);
    reid_config.LoadInteger(reid_name, "batch_size", config.batch_size);
    reid_config.LoadString(reid_name, "input_layer_name",
                           config.input_layer_name);

    reid_config.LoadBoolean(reid_name, "enable_fp16", config.enable_fp16);
    reid_config.LoadBoolean(reid_name, "enable_tf32", config.enable_tf32);
    reid_config.LoadBoolean(reid_name, "swapRB", config.swap_rb);

    config.input_layer_dimensions =
            reid_config.GetList<int>(reid_name, "input_layer_dimensions");
    config.output_layer_names =
            reid_config.GetList<std::string>(reid_name, "output_layer_names");

    return config;
}
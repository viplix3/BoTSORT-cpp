#pragma once

#include <string>
#include <vector>

struct ReIDParams
{
    long gpu_id{0};
    std::string distance_metric{"euclidean"};
    long trt_logging_level{1};
    long batch_size{1};
    std::string input_layer_name{""};
    std::vector<int> input_layer_dimensions;
    std::vector<std::string> output_layer_names;

    bool enable_fp16{true};
    bool enable_tf32{true};
    bool swap_rb{false};

    static ReIDParams load_config(const std::string &config_path);
};

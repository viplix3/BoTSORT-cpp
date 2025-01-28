#pragma once

#include <string>

struct TrackerParams
{
    bool reid_enabled{false};
    bool gmc_enabled{false};
    float track_high_thresh{0.6F};
    float track_low_thresh{0.1F};
    float new_track_thresh{0.7F};
    long track_buffer{30};
    float match_thresh{0.7F};
    float proximity_thresh{0.5F};
    float appearance_thresh{0.25F};
    std::string gmc_method_name{"sparseOptFlow"};
    long frame_rate{30};
    float lambda{0.985F};

    static TrackerParams load_config(const std::string &config_path);
};

#include "TrackerParams.h"

#include <iostream>

#include "DataType.h"
#include "INIReader.h"


TrackerParams TrackerParams::load_config(const std::string &config_path)
{
    TrackerParams config{};

    const std::string tracker_name = "BoTSORT";

    INIReader tracker_config(config_path);
    if (tracker_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    tracker_config.LoadBoolean(tracker_name, "enable_reid",
                               config.reid_enabled);
    tracker_config.LoadBoolean(tracker_name, "enable_gmc", config.gmc_enabled);
    tracker_config.LoadFloat(tracker_name, "track_high_thresh",
                             config.track_high_thresh);
    tracker_config.LoadFloat(tracker_name, "track_low_thresh",
                             config.track_low_thresh);
    tracker_config.LoadFloat(tracker_name, "new_track_thresh",
                             config.new_track_thresh);
    tracker_config.LoadInteger(tracker_name, "track_buffer",
                               config.track_buffer);
    tracker_config.LoadFloat(tracker_name, "match_thresh", config.match_thresh);
    tracker_config.LoadFloat(tracker_name, "proximity_thresh",
                             config.proximity_thresh);
    tracker_config.LoadFloat(tracker_name, "appearance_thresh",
                             config.appearance_thresh);
    tracker_config.LoadString(tracker_name, "gmc_method",
                              config.gmc_method_name);
    tracker_config.LoadInteger(tracker_name, "frame_rate", config.frame_rate);
    tracker_config.LoadFloat(tracker_name, "lambda", config.lambda);

    return config;
}
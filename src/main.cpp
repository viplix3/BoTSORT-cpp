
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

#include "BoTSORT.h"
#include "DataType.h"
#include "GlobalMotionCompensation.h"
#include "track.h"


void mot_format_writer(const std::vector<Track> &tracks, const std::string &output_file) {
    std::ofstream mot_file(output_file);
    for (const Track &track: tracks) {
        std::vector<float> bbox_tlwh = track.get_tlwh();
        float score = track.get_score();

        mot_file << track.frame_id << "," << track.track_id << "," << bbox_tlwh[0] << ","
                 << bbox_tlwh[1] << "," << bbox_tlwh[2] << "," << bbox_tlwh[3] << "," << score << ",-1,-1,-1" << std::endl;
    }
    mot_file.close();
}

std::vector<Detection> read_detections_from_file(const std::string &detection_file, int frame_width, int frame_height) {
    std::vector<Detection> detections;
    std::ifstream det_file(detection_file);
    std::string line;
    while (std::getline(det_file, line)) {
        std::istringstream iss(line);
        std::vector<float> values(std::istream_iterator<float>{iss}, std::istream_iterator<float>());

        Detection det;
        det.class_id = values[0];
        det.bbox_tlwh = cv::Rect_(values[1], values[2], values[3], values[4]);
        // bounding box is normalized, so convert to absolute coordinates
        det.bbox_tlwh.x *= frame_width;
        det.bbox_tlwh.y *= frame_height;
        det.bbox_tlwh.width *= frame_width;
        det.bbox_tlwh.height *= frame_height;
        det.confidence = values[5];
        detections.push_back(det);
    }
    det_file.close();
    return detections;
}

void plot_tracks(cv::Mat &frame, std::vector<Detection> &detections, std::vector<Track> &tracks) {
    for (const auto &det: detections) {
        cv::rectangle(frame, det.bbox_tlwh, cv::Scalar(0, 0, 0), 1);
    }

    for (const auto &track: tracks) {
        std::vector<float> bbox_tlwh = track.get_tlwh();
        cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        cv::rectangle(frame, cv::Rect(bbox_tlwh[0], bbox_tlwh[1], bbox_tlwh[2], bbox_tlwh[3]), color, 2);
    }
}

int main(int argc, char **argv) {

    if (argc < 4) {
        std::cout << "Usage: ./multi_object_tracking <images_dir> <dir_containing_per_frame_detections> <dir_to_save_mot_format_output>" << std::endl;
        return -1;
    }

    std::string images_dir = argv[1];
    std::string detection_dir = argv[2];
    std::string output_dir = argv[3];

    // Setup output directories
    std::string output_dir_mot = output_dir + "/mot";
    std::string output_dir_img = output_dir + "/img";
    std::filesystem::create_directories(output_dir_mot);
    std::filesystem::create_directories(output_dir_img);


    // Read filenames in images dir
    std::vector<std::string> image_filepaths;
    for (const auto &entry: std::filesystem::directory_iterator(images_dir)) {
        image_filepaths.push_back(entry.path());
    }
    std::sort(image_filepaths.begin(), image_filepaths.end());


    // Initialize BoTSORT tracker with all the deault params
    // TODO: Load BoTSORT params from a config file
    BoTSORT tracker = BoTSORT();


    // Read detections and execute MultiObjectTracker
    for (const auto &filepath: image_filepaths) {
        std::string filename = filepath.substr(filepath.find_last_of('/') + 1);
        filename = filename.substr(0, filename.find_last_of('.'));
        std::string detection_file = detection_dir + "/" + filename + ".txt";
        std::string output_file_txt = output_dir_mot + "/" + filename + ".txt";
        std::string output_file_img = output_dir_img + "/" + filename + ".jpg";

        // Read image and detections
        cv::Mat frame = cv::imread(filepath);
        std::vector<Detection> detections = read_detections_from_file(detection_file, frame.cols, frame.rows);

        // Execute tracker
        std::vector<Track> tracks = tracker.track(detections, frame);

        // Outputs
        mot_format_writer(tracks, output_file_txt);

        plot_tracks(frame, detections, tracks);
        cv::imwrite(output_file_img, frame);
    }

    return 0;
}
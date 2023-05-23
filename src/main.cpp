
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "BoTSORT.h"
#include "DataType.h"
#include "GlobalMotionCompensation.h"
#include "track.h"


#define TEST_GMC 0
#define GT_AS_PREDS 0
#define YOLOv8_PREDS 1

/**
 * @brief Read detections from MOTChallenge format file
 * 
 * @param tracks Tracks
 * @param output_file Output file where the tracks will be written
 */
void mot_format_writer(const std::vector<std::shared_ptr<Track>> &tracks, const std::string &output_file) {
    std::ofstream mot_file(output_file, std::ios::app);
    for (const std::shared_ptr<Track> &track: tracks) {
        std::vector<float> bbox_tlwh = track->get_tlwh();

        mot_file << track->frame_id << "," << track->track_id << "," << bbox_tlwh[0] << ","
                 << bbox_tlwh[1] << "," << bbox_tlwh[2] << "," << bbox_tlwh[3] << ",-1,-1,-1,0" << std::endl;
    }
    mot_file.close();
}
/**
 * @brief Read detections from YOLOv8 format file
 * 
 * @param detection_file Detection file
 * @param frame_width Image width (used to convert normalized bounding box to absolute coordinates)
 * @param frame_height Image height (used to convert normalized bounding box to absolute coordinates)
 * @return std::vector<Detection> Detections
 */
std::vector<Detection> read_detections_from_file(const std::string &detection_file, int frame_width, int frame_height) {
    std::vector<Detection> detections;
    std::ifstream det_file(detection_file);
    std::string line;
    while (std::getline(det_file, line)) {
        std::istringstream iss(line);
        std::vector<float> values(std::istream_iterator<float>{iss}, std::istream_iterator<float>());

        Detection det;
        if (values[0] != 0)
            continue;
        det.class_id = static_cast<int>(values[0]);
        det.bbox_tlwh = cv::Rect_(values[1] - values[3] / 2, values[2] - values[4] / 2, values[3], values[4]);
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


std::vector<std::vector<Detection>> read_mot_gt_from_file(const std::string &gt_filepath) {
    // https://stackoverflow.com/questions/57678677/multiple-object-tracking-mot-benchmark-data-set-format-for-ground-truth-tracki
    // BB format is: frame_no,object_id,bb_left,bb_top,bb_width,bb_height,score,X,Y,Z

    std::vector<std::vector<Detection>> all_gt_for_curr_sequence;
    std::ifstream gt_file(gt_filepath);
    std::string line;
    while (std::getline(gt_file, line)) {
        std::istringstream iss(line);
        std::vector<float> values;

        while (std::getline(iss, line, ',')) {
            values.push_back(std::stof(line));
        }

        Detection det;
        int frame_id = static_cast<int>(values[0]);
        det.class_id = 0;// class_id is not provided in MOTChallenge format, so set it to 0 to indicate person
        det.bbox_tlwh = cv::Rect_(values[2], values[3], values[4], values[5]);
        det.confidence = static_cast<float>(values[6]) == 0 ? 1.0f : static_cast<float>(values[6]);

        while (all_gt_for_curr_sequence.size() < frame_id) {
            all_gt_for_curr_sequence.emplace_back();
        }
        all_gt_for_curr_sequence[frame_id - 1].push_back(det);
    }

    return all_gt_for_curr_sequence;
}


/**
 * @brief Plot tracks on the frame
 * 
 * @param frame Input frame
 * @param detections Detections
 * @param tracks Tracks
 */
void plot_tracks(cv::Mat &frame, std::vector<Detection> &detections, std::vector<std::shared_ptr<Track>> &tracks) {
    static std::map<int, cv::Scalar> track_colors;
    cv::Scalar detection_color = cv::Scalar(0, 0, 0);
    for (const auto &det: detections) {
        cv::rectangle(frame, det.bbox_tlwh, detection_color, 1);
    }

    for (const std::shared_ptr<Track> &track: tracks) {
        std::vector<float> bbox_tlwh = track->get_tlwh();
        cv::Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

        if (track_colors.find(track->track_id) == track_colors.end()) {
            track_colors[track->track_id] = color;
        } else {
            color = track_colors[track->track_id];
        }

        cv::rectangle(frame,
                      cv::Rect(static_cast<int>(bbox_tlwh[0]),
                               static_cast<int>(bbox_tlwh[1]),
                               static_cast<int>(bbox_tlwh[2]),
                               static_cast<int>(bbox_tlwh[3])),
                      color,
                      2);
        cv::putText(frame,
                    std::to_string(track->track_id),
                    cv::Point(static_cast<int>(bbox_tlwh[0]), static_cast<int>(bbox_tlwh[1])),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2);

        cv::rectangle(frame, cv::Rect(10, 10, 20, 20), detection_color, -1);
        cv::putText(frame, "Detection", cv::Point(40, 25), cv::FONT_HERSHEY_SIMPLEX, 0.75, detection_color, 2);
    }
}

int main(int argc, char **argv) {

    if (argc < 4) {
        std::cout << "Usage: ./bot-sort-tracker <images_dir> <dir_containing_per_frame_detections> <dir_to_save_mot_format_output>" << std::endl;
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


    // Initialize BoTSORT tracker with all the default params
    // TODO: Load BoTSORT params from a config file
    BoTSORT tracker = BoTSORT();


// // Initialize GlobalMotionCompensation
#if (TEST_GMC == 1)
    /*
        {"orb", GMC_Method::ORB},
        {"ecc", GMC_Method::ECC},
        {"sparseOptFlow", GMC_Method::SparseOptFlow},
        {"optFlowModified", GMC_Method::OptFlowModified},
        {"OpenCV_VideoStab", GMC_Method::OpenCV_VideoStab},
    */
    GlobalMotionCompensation gmc = GlobalMotionCompensation(GlobalMotionCompensation::GMC_method_map["OpenCV_VideoStab"], 2.0);
    cv::VideoCapture cap("/home/vipin/datasets/test_videos/0x100000A9_424_20220427_094317.mp4");
    cv::Mat frame;

    while (cap.read(frame)) {
        HomographyMatrix H = gmc.apply(frame, {});
        cv::Mat H_cv;
        cv::eigen2cv(H, H_cv);
        cv::Mat warped_frame;
        cv::warpPerspective(frame, warped_frame, H_cv, frame.size());

        cv::imshow("frame", frame);
        cv::imshow("warped", warped_frame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    return 0;
#endif


    int frame_counter = 0;
    double tracker_time_sum = 0, tracker_time_total = 0;
    std::string output_file_txt = output_dir_mot + "/all.txt";

#if (YOLOv8_PREDS == 1)
    // Read detections and execute MultiObjectTracker
    for (const auto &filepath: image_filepaths) {
        std::string filename = filepath.substr(filepath.find_last_of('/') + 1);
        filename = filename.substr(0, filename.find_last_of('.'));
        std::string detection_file = detection_dir + "/" + filename + ".txt";
        std::string output_file_img = output_dir_img + "/" + filename + ".jpg";

        // Read image and detections
        cv::Mat frame = cv::imread(filepath);
        std::vector<Detection> detections = read_detections_from_file(detection_file, frame.cols, frame.rows);

        // Execute tracker
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::shared_ptr<Track>> tracks = tracker.track(detections, frame);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        tracker_time_sum += elapsed.count();

        // Outputs
        mot_format_writer(tracks, output_file_txt);

        plot_tracks(frame, detections, tracks);
        cv::imwrite(output_file_img, frame);

        frame_counter++;

        if (frame_counter % 100 == 0) {
            std::cout << "Processed " << frame_counter << " frames\t";
            std::cout << "Tracker FPS (last 100 frames): " << 100 / tracker_time_sum << std::endl;
            tracker_time_total += tracker_time_sum;
            tracker_time_sum = 0;
        }

        // if (frame_counter == 10)
        //     break;
    }
#endif


#if (GT_AS_PREDS == 1)
    std::vector<std::vector<Detection>> gt_per_frame = read_mot_gt_from_file(detection_dir);

    for (const auto &filepath: image_filepaths) {
        std::string filename = filepath.substr(filepath.find_last_of('/') + 1);
        filename = filename.substr(0, filename.find_last_of('.'));
        std::string output_file_img = output_dir_img + "/" + filename + ".jpg";

        // Read image and detections
        cv::Mat frame = cv::imread(filepath);

        // Execute tracker
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::shared_ptr<Track>> tracks = tracker.track(gt_per_frame[frame_counter], frame);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        tracker_time_sum += elapsed.count();

        // Outputs
        mot_format_writer(tracks, output_file_txt);

        plot_tracks(frame, gt_per_frame[frame_counter], tracks);
        cv::imwrite(output_file_img, frame);

        frame_counter++;

        if (frame_counter % 100 == 0) {
            std::cout << "Processed " << frame_counter << " frames\t";
            std::cout << "Tracker FPS (last 100 frames): " << 100 / tracker_time_sum << std::endl;
            tracker_time_total += tracker_time_sum;
            tracker_time_sum = 0;
        }
    }
#endif

    std::cout << "Average tracker FPS: " << frame_counter / tracker_time_total << std::endl;
    std::cout << "Average processing time per frame (ms): " << (tracker_time_total / frame_counter) * 1000 << std::endl;

    return 0;
}
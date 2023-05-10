#include "DataType.h"
#include "GlobalMotionCompensation.h"

#include <opencv2/core/eigen.hpp>

int main() {

    GlobalMotionCompensation gmc(GMC_Method::ORB, 2.0);

    // Apply GMC on video and show the result
    cv::VideoCapture cap("/home/vipin/datasets/test_videos/0x100000A9_424_20220427_094317.mp4");
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        std::vector<Detection> detections;
        HomographyMatrix H = gmc.apply(frame, detections);
        cv::Mat H_cvMat;
        eigen2cv(H, H_cvMat);

        cv::Mat frame_gmc;
        cv::warpPerspective(frame, frame_gmc, H_cvMat, frame.size());

        // cv::imshow("Frame", frame);
        // cv::imshow("Frame GMC", frame_gmc);

        if (cv::waitKey(25) == 27) {
            break;
        }
    }

    return 0;
}
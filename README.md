# BoT-SORT C++

> [**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**](https://arxiv.org/abs/2206.14651)
>
> Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky

## Introduction

This repository contains unofficial implementation of BoT-SORT tracker in C++.

[YOLOv8-tracking](https://github.com/mikel-brostrom/yolov8_tracking), [Official BoT-SORT](https://github.com/NirAharon/BoT-SORT) and [Official ByteTrack](https://github.com/ifzhang/ByteTrack) GitHub repositories were used as references.

## TODO

- [x] Implement BoT-SORT tracker
- [ ] Load parameters from config file
- [ ] Implement Re-ID model for BoT-SORT tracker using TensorRT

## Installation

Step 1. Install OpenCV

    Follow [this](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7) guide for OpenCV installation

Step 2. Install Eigen3

    ```bash
    sudo apt-get install libeigen3-dev
    ```

Step 3. Install CMake > 3.20

    ```bash
    sudo snap install cmake --classic
    ```

Step 4. Build BoT-SORT tracker

    ```bash
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make
    ```

## Usage

    ```bash
    ./bot-sort-tracker <images_dir> <dir_containing_per_frame_detections> <dir_to_save_mot_format_output>
    ```

# Performance Analysis Report

## Test settings

- MOT-20 train dataset, sequence MOT20-01 having 429 frames was used in testing.
- Code execution speed (ms or FPS) reported using the ‘Release’ build should be taken into consideration.
- Debug code build has been used for analysis and improvement purposes.
- Relative speed comparison, i.e. percentage of total execution time taken is done in Debug build.

## Profiling Results

### Execution time of different modules

| Build Type | Re-ID | Camera Motion Estimation | Motion Compensation | Kalman Filter | Algorithm Execution Time (ms) | Algorithm Execution FPS | Removed Module | Estimated time for the removed module |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Release | ❌ | ✅ | ✅ | ✅ | 4.9432 | 202.31 | N/A | N/A |
| Release | ❌ | ❌ | ✅ | ✅ | 0.0580 | 17242.36 | Camera Motion Estimation | 4.9432 - 0.0580 = 4.8852 |
| Release | ❌ | ❌ | ❌ | ✅ | 0.0546 | 18312.23 | Motion Compensation | 0.0580 - 0.0546 = 0.0034 |
| Debug | ❌ | ✅ | ✅ | ✅ | 8.9922 | 111.20 | N/A | N/A |
| Debug | ❌ | ❌ | ✅ | ✅ | 3.7505 | 266.62 | Camera Motion Estimation | 8.9922 - 3.7505 = 5.2417 |
| Debug | ❌ | ❌ | ❌ | ✅ | 3.1149 | 321.03 | Motion Compensation | 3.7505 - 3.1149 = 0.6356 |

- Re-ID: This is the CNN used for visual feature extraction. It has not been implemented in C++ codebase.
- Camera Motion Estimation: uses sparse optical flow to find homography matrix between previous and current frame of the video sequence.
- Motion Compensation: Applies motion compensation using the homography matrix predicted by camera motion estimation algorithm.
- Kalman Filter: State prediction algorithm used in the BoT-SORT tracking algorithm.

#### Low confidence (high number of bounding boxes/frame) performance analysis

For the MOT20-01 test sequence:

- Original settings (used for the table above) produces: ~ 15 tracks/frame.
- Low confidence settings produces: ~62 tracks/frame.

The table below shows tracking algorithm performance in the case of ~62 tracks/frame:

| Build Type | Re-ID | Camera Motion Estimation | Motion Compensation | Kalman Filter | Algorithm Execution Time (ms) | Algorithm Execution FPS | Removed Module | Estimated time for the removed module |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Release | ❌ | ✅ | ✅ | ✅ | 4.8348 | 206.83 | N/A | N/A |
| Release | ❌ | ❌ | ✅ | ✅ | 0.1590 | 6289.36 | Camera Motion Estimation | 4.8348 - 0.1590 = 4.6758 |
| Release | ❌ | ❌ | ❌ | ✅ | 0.1506 | 6647.40 | Motion Compensation | 0.1590 - 0.1506 = 0.0084 |

## Conclusions

- The Camera Motion Estimation function is the most time-consuming part of the program, both in the Release and Debug builds. In the Release build, it occupies 97.1% of the total execution time.
- Despite having a significant role in the tracking algorithm, the Kalman Filter function takes minimal time for execution. This is consistent across both Release and Debug builds.
- A variation in the number of boxes being tracked due to confidence thresholds significantly impacts the performance of the tracking algorithm. This was tested using lower confidence thresholds resulting in a higher number of bounding box predictions per frame, giving an approximation of the worst-case scenario performance.
- In the simulated worst-case scenario, with ~62 tracks/frame, Camera Motion Estimation still consumes the majority of execution time.
- Future optimizations, if necessary, can consider focusing on the Camera Motion Estimation function due to its large impact on overall performance.

## Accuracy Analysis

To evaluate the accuracy of the BoT-SORT tracker, the Multiple Object Tracking (MOT) metrics were employed. The MOT metrics were calculated using a script adopted from [py-motmetrics](https://github.com/cheind/py-motmetrics), a Python project dedicated for benchmarking multiple object trackers.

### Data Used for Analysis

The MOT Ground Truth provided by the MOT benchmark for 4 sequences (MOT-20-01, MOT-20-02, MOT-20-03, and MOT-20-05) was used as a prediction for both the Python and C++ tracking code. The Re-ID module was disabled in the Python version and a common Python script was used to calculate MOT metrics for both Python and C++ produced results.

### Key Results

| Input Sequence | Tracking Algorithm                                   | #Frames | #GT | #Predictions | MOTP  | MOTA  | IDF1   | ID-Switches | Precision | Recall | TP   | Predictions - FP | FP | FN |
|----------------|------------------------------------------------------|---------|-----|--------------|-------|-------|--------|-------------|-----------|--------|------|------------------|----|----|
| MOT-20-01      | BoT-SORT (without Re-ID CNN): Reference Python project   | 429     | 26647 | 26610        | 1.1175| 99.8611| 99.3184 | 0           | 100.00   | 99.8611| 26610 | 0                | 0  | 37 |
| MOT-20-01      | BoT-SORT (without Re-ID CNN): C++ implementation        | 429     | 26647 | 26610        | 3.5014| 99.8536| 98.8189 | 2           | 100.00   | 99.8611| 26610 | 0                | 0  | 37 |
| MOT-20-02      | BoT-SORT (without Re-ID CNN): Reference Python project   | 2782    | 202215 | 201973     | 0.7853| 99.8803| 98.4344 | 0           | 100.00   | 99.8803| 201973| 0                | 0  | 242 |
| MOT-20-02      | BoT-SORT (without Re-ID CNN): C++ implementation        | 2782    | 202215 | 201972     | 2.6858| 99.8630| 97.9408 | 20          | 99.9965  | 99.8764| 201965| 7                | 7  | 250 |
| MOT-20-03      | BoT-SORT (without Re-ID CNN): Reference Python project   | 2405    | 356728 | 356003     | 1.0984| 99.7892| 99.1364 | 27          | 100.00   | 99.7968| 356003| 0                | 0  | 725 |
| MOT-20-03      | BoT-SORT (without Re-ID CNN): C++ implementation        | 2405    | 356728 | 356069     | 2.3445| 99.7881| 99.6073 | 27          | 99.9902  | 99.8055| 356034| 35               | 35 | 694 |
| MOT-20-05      | BoT-SORT (without Re-ID CNN): Reference Python project   | 3315    | 751330 | 750247     | 0.7992| 99.8509| 99.8243 | 37          | 100.00   | 99.8559| 750247| 0                | 0  | 1083 |
| MOT-20-05      | BoT-SORT (without Re-ID CNN): C++ implementation        | 3315    | 751330 | 750345     | 2.2765| 99.8644| 99.6920 | 18          | 99.9989  | 99.8678| 750337| 8                | 8  | 993 |

It is observed that the MOTP (Multiple Object Tracking Precision) is quite low for both the Python and C++ implementations of the tracker. According to the literature, higher numbers indicate better accuracy for MOTP. The reason for the low MOTP score in this case is due to the definition of MOTP used in py-motmetrics. It is calculated as average distance over number of assigned objects. To convert this to a percentage, like in the MOTChallenge benchmarks, we would compute `(1 - MOTP) * 100`.

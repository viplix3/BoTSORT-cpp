# BoT-SORT

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

## Performance Analysis

Performance of the BoT-SORT tracker was evaluated and the key results are shown below:

### Release Build

| Removed Module              | Algorithm Execution Time (ms) | Algorithm Execution FPS |
|-----------------------------|-------------------------------|-------------------------|
| N/A                         | 4.9432                        | 202.31                  |
| Camera Motion Estimation    | 0.0580                        | 17242.36                |
| Motion Compensation         | 0.0546                        | 18312.23                |

Approximated worst case (with ~62 tracks/frame):

| Removed Module              | Algorithm Execution Time (ms) | Algorithm Execution FPS |
|-----------------------------|-------------------------------|-------------------------|
| N/A                         | 4.8348                        | 206.83                  |
| Camera Motion Estimation    | 0.1590                        | 6289.36                 |
| Motion Compensation         | 0.1506                        | 6647.40                 |

The 'Removed Module' column represents which module was disabled for that particular run.

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

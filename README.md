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

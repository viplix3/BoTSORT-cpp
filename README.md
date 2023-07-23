# BoT-SORT: Unofficial C++ Implementation

> [**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**](https://arxiv.org/abs/2206.14651)
>
> Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky

## Description

This repository contains an unofficial C++ implementation of BoT-SORT multi-pedestrian tracking algorithm.

This implementation has been tested **on NVIDIA Jetson NX and it achieves real-time performance** on 1080p videos.

[YOLOv8-tracking](https://github.com/mikel-brostrom/yolov8_tracking), [Official BoT-SORT](https://github.com/NirAharon/BoT-SORT) and [Official ByteTrack](https://github.com/ifzhang/ByteTrack) GitHub repositories were used as references.

## TODO

- [x] Implement BoT-SORT tracker
- [x] Load parameters from config file
- [ ] Implement Re-ID model for BoT-SORT tracker using TensorRT

## Preview of Results

These results demonstrate the BoT-SORT tracker, implemented in this repository, utilizing MOT provided detections.

[![MOT20-01](assets/MOT20-01.gif)](assets/MOT20-01.gif)

[More Result Videos](https://www.youtube.com/watch?v=elQrRxFKxno&list=PL4tNOKTSVsFArSjEoLXLNywmmIbQTvBs1)

## Installation Guide

1. OpenCV Installation

    Follow [this](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7) guide for OpenCV installation

2. Eigen3 Installation

    ```bash
    sudo apt install libeigen3-dev
    ```

3. Install CMake > 3.20

    ```bash
    sudo snap install cmake --classic
    ```

4. Build BoT-SORT tracker

    ```bash
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make
    ```

## Usage

Example code for using the BoT-SORT tracker is provided in [botsort_tracking_example.cpp](examples/botsort_tracking_example.cpp)
The project by default builds the example code.

To test the example code, run the following command:

```bash
cd <project-root-dir>/build
./bin/botsort_tracking_example ../config ../examples/data/MOT20-01.mp4 ../examples/data/det/det.txt ../output/
```

## Performance Analysis

The performance of the BoT-SORT tracker, implemented in this repository, was evaluated on the MOT20 dataset.
Details are provided in [this](docs/PerformanceReport.md) document.

### **Execution time of different modules (Host Machine, Release Build, Best of 3)**

| Sequence | Average Objects/Frame | Re-ID | Camera Motion Estimation | Motion Compensation | Kalman Filter | Algorithm Execution Time (ms) | Algorithm Execution FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOT20-01 | 62 | ❌ | ✅ | ✅ | ✅ | 4.7771 | 209.3333 |
| MOT20-01 | 62 | ❌ | ❌ | ✅ | ✅ | 0.2074 | 4819.2466 |
| MOT20-01 | 62 | ❌ | ❌ | ❌ | ✅ | 0.2016 | 4959.9266 |
| MOT20-05 | 226 | ❌ | ✅ | ✅ | ✅ | 6.6817 | 149.6633 |
| MOT20-05 | 226 | ❌ | ❌ | ✅ | ✅ | 2.0498 | 487.9570 |
| MOT20-05 | 226 | ❌ | ❌ | ❌ | ✅ | 2.0054 | 498.6453 |

### **Execution time of different modules (Jetson-NX, Release Build, Best of 3)**

| Sequence | Avg Objects/image | Re-ID | Camera Motion Estimation | Motion Compensation | Kalman Filter | Algorithm Execution Time (ms) | Algorithm Execution FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOT20-01 | 62 | ❌ | ✅ | ✅ | ✅ | 39.9968 | 25.0229 |
| MOT20-01 | 62 | ❌ | ❌ | ✅ | ✅ | 2.6441 | 378.2070 |
| MOT20-01 | 62 | ❌ | ❌ | ❌ | ✅ | 2.5185 | 397.0786 |
| MOT20-05 | 226 | ❌ | ✅ | ✅ | ✅ | 52.4251 | 19.0749 |
| MOT20-05 | 226 | ❌ | ❌ | ✅ | ✅ | 15.6337 | 63.9844 |
| MOT20-05 | 226 | ❌ | ❌ | ❌ | ✅ | 14.7918 | 67.7275 |

## Accuracy Analysis

The accuracy of the BoT-SORT tracker, implemented in this repository, was evaluated on the MOT20 train set.
Details are provided in [this](docs/AccuracyReport.md) document.

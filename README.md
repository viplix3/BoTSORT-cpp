# BoT-SORT: Unofficial C++ Implementation

> [**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**](https://arxiv.org/abs/2206.14651)
>
> Nir Aharon, Roy Orfaig, Ben-Zion Bobrovsky

## Description

This repository contains an unofficial C++ implementation of BoT-SORT multi-pedestrian tracking algorithm.

[YOLOv8-tracking](https://github.com/mikel-brostrom/yolov8_tracking), [Official BoT-SORT](https://github.com/NirAharon/BoT-SORT) and [Official ByteTrack](https://github.com/ifzhang/ByteTrack) GitHub repositories were used as references.

## TODO

- [x] Implement BoT-SORT tracker
- [ ] Load parameters from config file
- [ ] Implement Re-ID model for BoT-SORT tracker using TensorRT

## Preview of Results

These results demonstrate the BoT-SORT tracker, implemented in this repository, utilizing MOT provided detections.

[![MOT20-01](assets/MOT20-01.gif)](MOT20-01)

[More Result Videos](assets/)

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

To run the tracker on a video sequence, use the following command:

```bash
./bot-sort-tracker <images_dir> <dir_containing_per_frame_detections> <dir_to_save_mot_format_output>
```

## Performance

The performance of the BoT-SORT tracker, implemented in this repository, was evaluated on the MOT20 dataset.
Details are provided in [this](docs/PerformanceReport.md) document.

## Accuracy

The accuracy of the BoT-SORT tracker, implemented in this repository, was evaluated on the MOT20 train set.
Details are provided in [this](docs/AccuracyReport.md) document.

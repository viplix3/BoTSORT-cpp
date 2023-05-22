# Performance Analysis Report

## Test settings

- MOT-20 train dataset, sequence MOT20-01 having 429 frames was used in testing
- Performance is evaluated considering the execution speed (ms or FPS) of the 'Release' build

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

### Low confidence (high number of bounding boxes/frame) performance analysis

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

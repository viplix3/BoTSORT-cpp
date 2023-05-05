import argparse

import cv2
from tqdm import tqdm


def arg_parser():
    parser = argparse.ArgumentParser(description="Video VStack")
    parser.add_argument(
        "--source1",
        type=str,
        default="/home/vipin/GitRepos/MultiObjectTracking/yolov8_tracking/runs/track/exp/0x100000A9_424_20220427_094317.mp4",
        help="video 1 path (top)",
    )
    parser.add_argument(
        "--source1_name",
        type=str,
        default="Detector: YOLOv8-x (COCO)    Tracker: BoTSORT (with ReID)    Python Framework",
        help="video 2 path (bottom)",
    )

    parser.add_argument(
        "--source2",
        type=str,
        default="/home/vipin/GitRepos/ObjectDetectors/yolov8/runs/detect/track/0x100000A9_424_20220427_094317.mp4",
        help="video 2 path (bottom)",
    )
    parser.add_argument(
        "--source2_name",
        type=str,
        default="Detector: YOLOv8-x (COCO)    Tracker: BoTSORT (no ReID)    Python Framework",
        help="video 2 name (bottom)",
    )

    parser.add_argument("--output", type=str, required=True, help="output video path")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    source1_vid = cv2.VideoCapture(args.source1)
    source2_vid = cv2.VideoCapture(args.source2)

    vid1_len = int(source1_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid2_len = int(source2_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    destvid_len = min(vid1_len, vid2_len)

    source1_fps = source1_vid.get(cv2.CAP_PROP_FPS)
    source2_fps = source2_vid.get(cv2.CAP_PROP_FPS)

    source1_width = int(source1_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    source1_height = int(source1_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source2_width = int(source2_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    source2_height = int(source2_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dest_width = max(source1_width, source2_width)
    dest_height = source1_height + source2_height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, source1_fps, (dest_width, dest_height))

    for _ in tqdm(range(destvid_len)):
        ret1, frame1 = source1_vid.read()
        ret2, frame2 = source2_vid.read()

        if not ret1 or not ret2:
            break

        frame1 = cv2.putText(
            frame1,
            args.source1_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        frame2 = cv2.putText(
            frame2,
            args.source2_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        frame1 = cv2.resize(frame1, (dest_width, dest_height // 2))
        frame2 = cv2.resize(frame2, (dest_width, dest_height // 2))

        frame = cv2.vconcat([frame1, frame2])
        out.write(frame)

    source1_vid.release()
    source2_vid.release()
    out.release()


if __name__ == "__main__":
    args = arg_parser()
    main(args)

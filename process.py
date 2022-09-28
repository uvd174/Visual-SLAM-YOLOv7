import argparse
import os.path
import subprocess
import shutil
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from modules.video_reader import VideoReader


def draw_axes(img, H: np.ndarray, points: np.ndarray):
    if H is not None:
        points = cv2.perspectiveTransform(points, H)

    draw_points = points.reshape((-1, 1, 2)).astype(np.int32)

    img = cv2.line(img, tuple(draw_points[0].ravel()), tuple(draw_points[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, tuple(draw_points[2].ravel()), tuple(draw_points[3].ravel()), (0, 255, 0), 3)
    return img, points


def load_boxes(boxes_path: str, frame_width: int, frame_height: int) -> List[List[float]]:
    boxes = []

    try:
        with open(boxes_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                box = [float(x) for x in line.split(' ')][1:]

                box[2] += box[0]
                box[3] += box[1]

                box[0] *= frame_width
                box[1] *= frame_height
                box[2] *= frame_width
                box[3] *= frame_height

                boxes.append(box)
    except FileNotFoundError:
        pass

    return boxes


def lies_in_box(point: Tuple[float, float], box: List[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='raw_videos/golf_test.mp4', help='input video path')
    parser.add_argument('--output_dir', type=str, default='processed_videos', help='output video path')
    parser.add_argument('--device', default='cpu', help='cpu | device_id')
    opt = parser.parse_args()
    filename = os.path.basename(opt.input)

    # Start with Human Detection
    process_command = f'python yolov7/detect.py --weights yolov7.pt --source {opt.input} --device {opt.device}' \
                      f' --project {opt.output_dir} --name . --save-txt --no-trace --exist-ok --class 0 --nosave'

    shutil.rmtree(os.path.join(opt.output_dir, 'labels'), ignore_errors=True)

    subprocess.run(process_command, shell=True, text=True)

    MIN_MATCH_COUNT = 15

    frame_provider = VideoReader(opt.input, mode='GRAY')

    prev_frame = next(frame_provider)
    prev_boxes = load_boxes(
        os.path.join(opt.output_dir, 'labels', filename.replace('.mp4', '_1.txt')),
        frame_provider.width, frame_provider.height,
    )

    homographies = []

    for index, new_frame in enumerate(tqdm(frame_provider, desc='Processing frames')):
        new_frame = new_frame
        new_boxes = load_boxes(
            os.path.join(opt.output_dir, 'labels', filename.replace('.mp4', f'_{index + 2}.txt')),
            frame_provider.width, frame_provider.height,
        )

        # Applying the function
        sift = cv2.SIFT_create()

        prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
        new_kp, new_des = sift.detectAndCompute(new_frame, None)

        # filter Feature Points by human boxes
        prev_kp_des = zip(prev_kp, prev_des)
        prev_kp_des = [
            (kp, des) for kp, des in prev_kp_des
            if not any([lies_in_box(kp.pt, box) for box in prev_boxes])
        ]

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(prev_des, new_des, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([prev_kp[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([new_kp[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)

            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            homographies.append(homography)
        else:
            print("Frame {}, not enough matches are found - {}/{}".format(index, len(good), MIN_MATCH_COUNT))
            homographies.append(None)

        prev_frame = new_frame
        prev_boxes = new_boxes

    frame_provider = VideoReader(opt.input)

    video_writer = cv2.VideoWriter(
        os.path.join(opt.output_dir, filename),
        cv2.VideoWriter_fourcc(*'mp4v'),
        frame_provider.fps,
        (frame_provider.width, frame_provider.height),
    )

    w, h = frame_provider.height, frame_provider.width
    points = np.float32([
        [h // 2, 0], [h // 2, w], [0, w // 2], [h, w // 2]
    ]).reshape(-1, 1, 2)

    # Visualize the homographies
    for i, homography_frame in enumerate(
            tqdm(zip(homographies, frame_provider), total=len(homographies), desc='Visualizing')):
        homography, frame = homography_frame

        frame, points = draw_axes(frame, homography, points)

        video_writer.write(frame)

    video_writer.release()

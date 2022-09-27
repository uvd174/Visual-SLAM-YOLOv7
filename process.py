import argparse
import os.path
import subprocess
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from modules.video_reader import VideoReader


def draw_axes(img, H: np.ndarray, points: np.ndarray):
    # unit is mm
    if H is not None:
        points = cv2.perspectiveTransform(points, H)

    draw_points = points.reshape((-1, 1, 2)).astype(np.int32)

    img = cv2.line(img, tuple(draw_points[0].ravel()), tuple(draw_points[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, tuple(draw_points[2].ravel()), tuple(draw_points[3].ravel()), (0, 255, 0), 3)
    return img, points


def load_boxes(boxes_path, frame_index):
    real_path = boxes_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='raw_videos/golf_test.mp4', help='input video path')
    parser.add_argument('--output_dir', type=str, default='processed_videos', help='output video path')
    parser.add_argument('--device', default='cpu', help='cpu | device_id')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    opt = parser.parse_args()
    filename = os.path.basename(opt.input)

    # Start with Human Detection
    process_command = f'python yolov7/detect.py --weights yolov7/yolov7.pt --source {opt.input} --device {opt.device}' \
                      f' --project {opt.output_dir} --name . --save-txt --no-trace --exist-ok --class 0 --nosave'

    shutil.rmtree(os.path.join(opt.output_dir, 'labels'), ignore_errors=True)

    subprocess.run(process_command, shell=True, text=True)

    MIN_MATCH_COUNT = 15

    video_iterator = iter(VideoReader(opt.input))

    prev_frame = next(video_iterator)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_boxes = load_boxes(os.path.join(opt.output_dir, 'labels', filename.replace('.mp4', '.txt')), 0)

    homographies = []

    for index, new_frame in enumerate(tqdm(video_iterator)):
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Applying the function
        sift = cv2.ORB_create()

        prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
        new_kp, new_des = sift.detectAndCompute(new_frame, None)

        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)

        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.match(prev_des, new_des)

        # Extract location of matches
        prev_points = np.zeros((len(matches), 2), dtype=np.float32)
        new_points = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            prev_points[i, :] = prev_kp[match.queryIdx].pt
            new_points[i, :] = new_kp[match.trainIdx].pt

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([prev_kp[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([new_kp[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)

            homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            homographies.append(homography)
        else:
            print("Frame {}, not enough matches are found - {}/{}".format(index, len(good), MIN_MATCH_COUNT))
            homographies.append(None)

        prev_frame = new_frame

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
    for i, homography_frame in enumerate(tqdm(zip(homographies, frame_provider), total=len(homographies))):
        homography, frame = homography_frame

        frame, points = draw_axes(frame, homography, points)

        video_writer.write(frame)

    video_writer.release()

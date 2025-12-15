from pathlib import Path

import cv2
import numpy as np
from traditional_pipeline import TraditionalDetector
import argparse
import timeit
import utils

WNAME = "Video"

points = []

def on_mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append(np.array([x, y]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", type=Path, required=True)
    parser.add_argument("--board", "-b", type=Path, required=True)
    args = parser.parse_args()

    video = utils.load_video(args.video)
    board_img = cv2.imread(str(args.board))
    assert board_img is not None

    board_img = cv2.resize(board_img, utils.IMAGE_SIZE)
    detector = TraditionalDetector(board_img, buffer_length=3)
    lines = []

    for frame in utils.iter_video(video):
        processed, line = detector.process_frame(frame, 0.4)
        if line is not None:
            lines.append(line)

        for x0, y0, x1, y1 in lines:
            cv2.line(processed, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)

        for p in detector.confirmed_hits:
            cv2.drawMarker(
                processed,
                p,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                10,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WNAME, processed)

        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            break

    cv2.setMouseCallback(WNAME, on_mouse_cb)
    cv2.imshow(WNAME, processed)
    cv2.waitKey(0)

    print(detector.confirmed_hits)
    print(points)
    dists = np.linalg.norm(np.array(points) - np.array(detector.confirmed_hits), axis=1)
    print(f"Found darts: {len(detector.confirmed_hits)}/3 ({len(detector.confirmed_hits)/3})")
    print(f"Mean distance in pixels: {dists.mean()}")

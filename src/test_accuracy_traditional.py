from pathlib import Path

import cv2
from traditional_pipeline import TraditionalDetector
import argparse

import utils

WNAME = "Video"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", type=Path, required=True)
    parser.add_argument("--board", "-b", type=Path, required=True)
    args = parser.parse_args()

    video = utils.load_video(args.video)
    board_img = cv2.imread(str(args.board))
    assert board_img is not None

    board_img = cv2.resize(board_img, utils.IMAGE_SIZE)
    detector = TraditionalDetector(board_img)
    lines = []

    for frame in utils.iter_video(video):
        processed, line = detector.process_frame(frame)
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

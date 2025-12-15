from pathlib import Path

import cv2
from ml_pipeline import MLDetector
import argparse

import utils

WNAME = "Video"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", "-v", type=Path, required=True)
    parser.add_argument("--model", "-m", type=Path, required=True)
    args = parser.parse_args()

    video = utils.load_video(args.video)
    detector = MLDetector(
        args.model,
        {0: "D11_14", 1: "D19_3", 2: "D20_1", 3: "D6_10", 4: "dart_hit"},
        eps=10,
        min_samples=5,
    )

    for frame in utils.iter_video(video):
        detector.process_frame(frame)

        if detector.homography_H is not None:
            frame = cv2.warpPerspective(frame, detector.homography_H, utils.IMAGE_SIZE)

        for p in detector.confirmed_hits:
            cv2.drawMarker(
                frame,
                p,
                (0, 0, 255),
                cv2.MARKER_CROSS,
                10,
                2,
                cv2.LINE_AA,
            )
            
        cv2.imshow(WNAME, frame)

        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            break
    
    for p in detector.confirmed_hits:
        print(utils.score_dart(p))

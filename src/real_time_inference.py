import pathlib
from model_wrapper import CordCNN
from pathlib import Path
import time
from typing import Callable, List
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
import logging
from collections import deque
from ultralytics import YOLO

DISPLAY_SIZE = 640

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.DEBUG
)


def calculate_homography(corners: dict) -> np.ndarray:
    r = (DISPLAY_SIZE / 2) - 100
    dst = [
        (r, np.deg2rad(180 - 9)),
        (r, np.deg2rad(270 - 9)),
        (r, np.deg2rad(90 - 9)),
        (r, np.deg2rad(360 - 9)),
    ]

    dst = [
        [
            DISPLAY_SIZE / 2 + int(v[0] * np.cos(v[1])),
            DISPLAY_SIZE / 2 - int(v[0] * np.sin(v[1])),
        ]
        for v in dst
    ]

    return cv2.findHomography(
        np.array(list(corners.values())),
        np.array(dst),
        cv2.RANSAC,
        10.0,
    )[0]


if __name__ == "__main__":
    label_dict = {
        0: "D11_14",
        1: "D19_3",
        2: "D20_1",
        3: "D6_10",
        4: "dart_hit",
    }

    corners = {
        "D11_14": np.array((0.0, 0.0)),
        "D19_3": np.array((0.0, 0.0)),
        "D20_1": np.array((0.0, 0.0)),
        "D6_10": np.array((0.0, 0.0)),
    }

    corner_eps = 5.0  # px
    root = Path(__file__).parent.parent.resolve()

    # cam, center_crop = utils.setup_camera(file=Path(r"G:\My Drive\IMG_1320.mov"))
    cam, center_crop = utils.setup_camera(file=Path(r"C:\Users\Martin\Videos\Camo Studio Recording 2025-12-14 17-52-28.mp4"))

    cv2.namedWindow("Dart")

    try:
        model = YOLO(root / "trained_models/yolov8n_darts/weights/best.pt")
    except NotImplementedError:
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        model = YOLO(root / "trained_models/yolov8n_darts/weights/best.pt")
        pathlib.PosixPath = temp

    H = np.zeros((3, 3), np.float32)

    darts = np.zeros(shape=(128, 3), dtype=np.float32)
    hit_count = 0
    
    while 1:
        ret, curr_color = cam.read()
        if not ret:
            break
        print(curr_color.shape)
        curr_color = cv2.resize(curr_color, (640, 640))
        res = model.predict(curr_color, verbose=False)[0]

        for idx, cls in enumerate(res.boxes.cls):
            name = label_dict[int(cls)]
            p = res.boxes.xywh[idx, :2].numpy()
            if name in corners:
                dist = np.linalg.norm(corners[name] - p)
                if dist > corner_eps:
                    corners[name] = p
                    H = calculate_homography(corners)
            else:
                dist = np.linalg.norm(darts[:, :2] - p, axis=1)
                idx = dist.argmin()
                if dist[idx] < 5.0 and darts[idx, -1] < 10:
                    darts[idx, :2] = (darts[idx, :2] + p) / 2
                    darts[idx, -1] += 1
                elif dist[idx] < 10.0:
                    pass
                else:
                    darts[hit_count, :2] = p
                    hit_count += 1

        curr_color_warp = cv2.warpPerspective(
            curr_color, H, (DISPLAY_SIZE, DISPLAY_SIZE)
        )
        
        mask = np.where(darts[:, -1] >= 10)[0]
        if len(mask) > 0:
            tmp = darts[mask, :2].reshape(len(mask), 1, 2)
            points = cv2.perspectiveTransform(tmp, H)[:,0]
            for p in points:
                cv2.drawMarker(
                    curr_color_warp,
                    p.astype(int),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    15,
                    2,
                )
        cv2.imshow("Dart", curr_color_warp)
        k = cv2.waitKey(1)
        if k % 0xFF in (27, ord("q")):
            break
        
        elif k % 0xFF == ord(" "):
            darts = np.zeros(shape=(128, 3), dtype=np.float32)
            hit_count = 0
    

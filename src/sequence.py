from collections import namedtuple
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.stats
import sys
import utils


def imgload(path: Path):
    img_color = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img_color is None:
        raise IOError("Couldn't load board image." + str(path))

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGRA2GRAY)

    return (img_color, img_gray)


def do_sift(board_img: cv2.typing.MatLike, scene_img: cv2.typing.MatLike):
    sift = cv2.SIFT.create()

    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_img, None)  # type: ignore
    board_keypoints, board_descriptors = sift.detectAndCompute(board_img, None)  # type: ignore

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore

    matches: List[cv2.DMatch] = [
        m
        for m, n in flann.knnMatch(scene_descriptors, board_descriptors, k=2)
        if m.distance < 0.8 * n.distance
    ]

    board_points = np.array(
        [board_keypoints[m.trainIdx].pt for m in matches], dtype=np.float32
    ).reshape(-1, 1, 2)
    scene_points = np.array(
        [scene_keypoints[m.queryIdx].pt for m in matches], dtype=np.float32
    ).reshape(-1, 1, 2)

    H_board = cv2.findHomography(
        board_points, scene_points, cv2.RANSAC, 10.0, confidence=0.99
    )[0]
    H_scene = cv2.findHomography(
        scene_points, board_points, cv2.RANSAC, 10.0, confidence=0.99
    )[0]
    return (H_board, H_scene)


if __name__ == "__main__":
    gamma = 0.7
    invGamma = 1.0 / gamma
    gamma_lut = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)], dtype=np.uint8
    )

    lut_sqrt = np.sqrt(np.arange(256, dtype=np.float32) * 255).astype(np.uint8)

    root = Path(__file__).parent.parent.resolve()

    board_path = root / r"data\sift\bg_removed\board1.png"
    board_color, board_gray = imgload(board_path)
    board_mask = board_color[:, :, 3] != 0

    board_mask = cv2.GaussianBlur(
        cv2.dilate(board_mask.astype(np.uint8) * 255, np.ones((3, 3))), (3, 3), 1
    ).clip(0, 128)
    cv2.threshold(board_mask, 0, 255, cv2.THRESH_BINARY, board_mask)
    board_color[~board_mask.astype(np.bool)] = (0, 0, 0, 0)
    board_gray[~board_mask.astype(np.bool)] = 0
    cv2.LUT(board_gray, lut_sqrt, board_gray)

    sequence_dir = root / "data/sequence/seq3"
    sequence = [imgload(k) for k in sequence_dir.glob("*.jpg")]

    bg_color, bg_gray = sequence.pop(0)
    cv2.LUT(bg_gray, lut_sqrt, bg_gray)
    cv2.GaussianBlur(bg_gray, (3, 3), 2, bg_gray)
    cv2.normalize(bg_gray, bg_gray, 255, 0, cv2.NORM_MINMAX)

    H_board, H_scene = do_sift(board_gray, bg_gray)
    # print(H_board)
    # print(H_scene)
    # utils.display_preview(bg_gray, board_gray, title="Setup")
    while 1:
        try:
            curr_color, curr_gray = sequence.pop(0)
            cv2.LUT(curr_gray, lut_sqrt, curr_gray)
            cv2.GaussianBlur(curr_gray, (3, 3), 2, curr_gray)
            cv2.normalize(curr_gray, curr_gray, 255, 0, cv2.NORM_MINMAX)
        except IndexError:
            break

        diff = cv2.absdiff(curr_gray, bg_gray)
        diff = cv2.LUT(diff, gamma_lut)
        cv2.normalize(diff, diff, 255, 0, cv2.NORM_MINMAX)

        diff[diff < np.quantile(diff, 0.99)] = 0
        cv2.dilate(diff, np.ones((3, 3)), diff)
        utils.display_preview(diff)
        canny_diff = cv2.Canny(diff, 100, 200)

        y, x = np.where(canny_diff)
        points = np.column_stack((x, y)).astype(np.float32)
        [vx], [vy], x0, y0 = cv2.fitLine(
            points, distType=cv2.DIST_FAIR, param=0.2, reps=0.01, aeps=0.01
        )

        k = float(vy / vx)
        d = float(y0[0] - k * x0[0])

        xmin = int(x.min())
        xmax = int(x.max())

        pt1 = (xmin, int(k * xmin + d))
        pt2 = (xmax, int(k * xmax + d))
        cv2.line(curr_color, pt1, pt2, (0, 0, 255), 1)
        # cv2.warpPerspective(curr_color, H_scene, curr_gray.shape, curr_color)

        utils.display_preview(curr_color)

        bg_color = curr_color
        bg_gray = curr_gray

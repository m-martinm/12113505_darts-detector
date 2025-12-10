from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import utils


def get_good_matches(
    flann_matcher: cv2.FlannBasedMatcher,
    query_descriptors: np.ndarray,
    threshold: float = 0.7,
) -> List[cv2.DMatch]:
    matches = flann_matcher.knnMatch(query_descriptors, k=2)
    good_matches = []

    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    return good_matches


def load_board_reference(path: Path):
    board_img = cv2.imread(str(board_path), cv2.IMREAD_UNCHANGED)
    if board_img is None:
        raise IOError("Couldn't load board image." + str(path))

    board_mask = board_img[:, :, 3] > 0
    board_img = cv2.cvtColor(board_img, cv2.COLOR_BGRA2GRAY)

    return (board_img, board_mask)


if __name__ == "__main__":
    root = Path(__file__).parent.parent.resolve()

    dart_references = list((root / "data/sift/bg_removed").glob("dart*.jpg"))
    dart_images = [cv2.imread(str(dr), cv2.IMREAD_GRAYSCALE) for dr in dart_references]

    board_path = root / "data/sift/bg_removed/board1.jpg"
    (board_img, board_mask) = load_board_reference(board_path)

    scene_path = root / "data/real_images/IMG_1284.jpg"
    scene_img = cv2.imread(str(scene_path), cv2.IMREAD_GRAYSCALE)

    if any([i is None for i in dart_images]) or board_img is None or scene_img is None:
        print("Couldn't load images")
        sys.exit(1)

    board_np = utils.cv2numpy(board_img)
    board_np /= board_np.max()
    board_img = utils.numpy2cv(board_np)

    scene_np = utils.cv2numpy(scene_img)
    scene_np /= scene_np.max()
    scene_img = utils.numpy2cv(scene_np)
    # utils.display_preview(board_img, scene_img)
    # exit(0)

    # cv2.norm

    sift = cv2.SIFT.create()

    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_img, None)  # type: ignore
    board_keypoints, board_descriptors = sift.detectAndCompute(board_img, None)  # type: ignore

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore
    flann.add([board_descriptors])
    flann.train()

    board_matches = get_good_matches(flann, scene_descriptors)
    board_points = np.float32(
        [board_keypoints[m.trainIdx].pt for m in board_matches]
    ).reshape(-1, 1, 2)
    scene_points = np.float32(
        [scene_keypoints[m.queryIdx].pt for m in board_matches]
    ).reshape(-1, 1, 2)

    H_board = cv2.findHomography(board_points, scene_points, cv2.RANSAC, 5.0)[0]

    # H_scene = cv2.findHomography(scene_points, board_points, cv2.RANSAC, 5.0)[0]
    # scene_transformed = cv2.warpPerspective(scene_img, H_scene, board_img.shape)

    board_mask_transformed = cv2.warpPerspective(
        np.uint8(board_mask) * 255, H_board, scene_img.shape
    )

    obj_img = cv2.warpPerspective(utils.numpy2cv(board_np), H_board, scene_img.shape)

    canny_scene = cv2.Canny(utils.numpy2cv(scene_np), 100, 200)
    canny_board = cv2.Canny(obj_img, 100, 200)

    # utils.display_preview(canny_scene, out)

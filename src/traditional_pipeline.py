from collections import deque
from typing import Optional, Tuple
import cv2
import numpy as np

import utils


class TraditionalDetector:
    def __init__(self, board_img: np.ndarray, buffer_length: int = 5) -> None:
        self.sift = cv2.SIFT.create()
        self.load_board_reference(board_img)
        self.background = None
        self.homography_H = None
        self.dart_mask = np.zeros(shape=utils.IMAGE_SIZE, dtype=np.bool)
        self.sliding_window = deque([0.0 for _ in range(50)], maxlen=50)
        self.buffer_length = buffer_length
        self.buffer_cursor = 0
        self.buffer = np.empty(shape=(self.buffer_length, 4), dtype=np.float32)
        self.confirmed_hits = []

    def reset(self):
        self.background = None
        self.homography_H = None
        self.dart_mask = np.zeros(shape=utils.IMAGE_SIZE, dtype=np.bool)
        self.buffer_cursor = 0
        self.buffer = np.empty(shape=(self.buffer_length, 4), dtype=np.float32)

    def load_board_reference(self, board_img: np.ndarray):
        self.board_img = board_img
        self.board_keypoints, self.board_descriptors = self.sift.detectAndCompute(
            board_img,
            None,  # type: ignore
            None,
            False,
        )

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore

    def calculate_homography(
        self, scene_img: np.ndarray, threshold: float = 0.7
    ) -> np.ndarray:
        scene_keypoints, scene_descriptors = self.sift.detectAndCompute(scene_img, None)  # type: ignore
        matches = [
            m
            for m, n in self.flann.knnMatch(
                scene_descriptors, self.board_descriptors, k=2
            )
            if m.distance < threshold * n.distance
        ]
        board_points = np.array(
            [self.board_keypoints[m.trainIdx].pt for m in matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        scene_points = np.array(
            [scene_keypoints[m.queryIdx].pt for m in matches], dtype=np.float32
        ).reshape(-1, 1, 2)

        H = cv2.findHomography(scene_points, board_points, cv2.RANSAC, 5.0)[0]

        return H

    def segment_dart(self, current: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.background is None:
            raise RuntimeError("Something went wrong, background image is None")

        diff = cv2.absdiff(self.background, current)
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        total_change = cv2.countNonZero(thresh)

        if total_change == 0:
            return 0.0, thresh
        kernel = np.ones((5, 5), np.uint8)
        eroded_img = cv2.erode(thresh, kernel, iterations=1)
        surviving_change = cv2.countNonZero(eroded_img)
        ratio = surviving_change / total_change

        return ratio, thresh

    def process_frame(
        self, frame: np.ndarray, threshold: float = 0.5
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        frame = cv2.resize(frame, utils.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        result = None
        if self.homography_H is None:
            self.homography_H = self.calculate_homography(frame)

        frame = cv2.warpPerspective(frame, self.homography_H, utils.IMAGE_SIZE)

        frame_color, frame_gray = utils.preprocess_frame(frame)

        if self.background is None:
            self.background = frame_gray
            return (frame_color, result)

        ratio, dart = self.segment_dart(frame_gray)

        if ratio > threshold:
            dart[self.dart_mask] = 0
            cv2.imshow("aa", dart)

            y, x = np.where(dart)

            yfit, residuals, k, d = utils.fit_line(x, y)
            sigma_residuals = np.std(residuals)
            inlier_mask = np.abs(residuals) < sigma_residuals * 2
            if inlier_mask.mean() > 0.90:
                # self.dart_mask[y, x] = 1
                xmin = x[inlier_mask].min()
                xmax = x[inlier_mask].max()
                self.buffer[self.buffer_cursor] = np.array(
                    [[xmin, yfit[-1], xmax, yfit[0]]], dtype=np.float32
                )
                self.buffer_cursor += 1
            else:
                self.buffer_cursor = 0
        else:
            self.buffer_cursor = 0

        if self.buffer_cursor >= self.buffer_length:
            self.buffer_cursor = 0
            p1x, p1y, p2x, p2y = self.buffer[2:].mean(
                axis=0
            )  # discard he first two, maybe the dart was still moving
            center = np.array(dart.shape[:2]) / 2
            d1 = np.linalg.norm(center - np.array((p1y, p1x)))
            d2 = np.linalg.norm(center - np.array((p2y, p2x)))
            if d1 < d2:
                self.confirmed_hits.append(np.array((p1x, p1y), dtype=np.int32))
            else:
                self.confirmed_hits.append(np.array((p2x, p2y), dtype=np.int32))

            result = self.buffer[2:].mean(axis=0, dtype=np.int32)
        if self.buffer_cursor == 0:
            self.background = frame_gray

        return (frame_color, result)

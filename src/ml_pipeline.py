from pathlib import Path
from typing import Dict, Optional
import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore
import utils


class MLDetector:
    def __init__(
        self,
        model_path: Path,
        label_map: Dict[int, str],
        eps: float = 5.0,
        min_samples: int = 10,
    ) -> None:
        self.model: YOLO = utils.load_model(model_path)
        self.label_map = label_map
        self.display_size = utils.IMAGE_SIZE
        self.eps = eps

        self.homography_H: Optional[np.ndarray] = None
        self.corner_coords: Dict[str, np.ndarray] = {
            "D20_1": np.zeros(2),
            "D6_10": np.zeros(2),
            "D19_3": np.zeros(2),
            "D11_14": np.zeros(2),
        }

        self.confirmed_hits = []
        self.maybe_hit = np.zeros(shape=(128, 3), dtype=np.float32)
        self.hit_count = 0
        self.min_samples = min_samples

    def reset(self):
        self.confirmed_hits = []
        self.maybe_hit = np.zeros(shape=(128, 3), dtype=np.float32)
        self.hit_count = 0

    def calculate_homography(self) -> np.ndarray:
        dst = utils.get_calibration_points()

        corner_keys = ["D20_1", "D6_10", "D19_3", "D11_14"]

        src_points = np.array(
            [self.corner_coords[k] for k in corner_keys], dtype=np.float32
        )
        dst_points = np.array([dst[k] for k in corner_keys], dtype=np.float32)
        return cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            5.0,
        )[0]

    def process_frame(self, frame: np.ndarray):
        result = utils.infer(self.model, frame)
        if result is None:
            return

        # Sort them so dart_hit (4) is always last
        classes, xywh = result
        sort_indices = np.argsort(classes)
        classes = classes[sort_indices]
        xywh = xywh[sort_indices]

        for cls, box in zip(classes, xywh):
            name = self.label_map[int(cls)]
            p = box[:2]
            if name in self.corner_coords:
                dist = np.linalg.norm(self.corner_coords[name] - p)
                if dist > self.eps:
                    self.corner_coords[name] = p
                    self.homography_H = self.calculate_homography()

            elif self.homography_H is not None:
                p_warped = cv2.perspectiveTransform(
                    p.reshape(1, 1, 2), self.homography_H
                )[0, 0]
                dist = np.linalg.norm(self.maybe_hit[:, :2] - p_warped, axis=1)
                idx = dist.argmin()
                if dist[idx] < self.eps and self.maybe_hit[idx, -1] < self.min_samples:
                    # Dart already registered
                    self.maybe_hit[idx, :2] = (self.maybe_hit[idx, :2] + p_warped) / 2
                    self.maybe_hit[idx, -1] += 1
                    if self.maybe_hit[idx, -1] >= self.min_samples:
                        # Permanent hit
                        self.confirmed_hits.append(self.maybe_hit[idx, :2])

                elif dist[idx] > 2 * self.eps:
                    if self.hit_count < self.maybe_hit.shape[0]:
                        # New dart
                        self.maybe_hit[self.hit_count, :2] = p_warped
                        self.hit_count += 1
                    else:
                        raise OverflowError()

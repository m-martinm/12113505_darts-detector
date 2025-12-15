import pathlib
from typing import Dict, Iterable, Optional, Tuple
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import torch
from ultralytics import YOLO  # type: ignore
import numpy as np

IMAGE_SIZE = (640, 640)

_CAL_PADDING = 100
_R = min(IMAGE_SIZE) / 2 - _CAL_PADDING
_CAL_POINTS = {
    "D20_1": np.array([_R, np.deg2rad(90 - 9)]),
    "D6_10": np.array([_R, np.deg2rad(360 - 9)]),
    "D19_3": np.array([_R, np.deg2rad(270 - 9)]),
    "D11_14": np.array([_R, np.deg2rad(180 - 9)]),
}

OUTER_BULL_RADIUS = _R * 16 / 170
INNER_BULL_RADIUS = _R * (12.7 / 2) / 170
TRIPLE_OUTER_RADIUS = _R * 107 / 170
TRIPLE_INNER_RADIUS = _R * (107 / 170 - 8 / 170)
DOUBLE_INNER_RADIUS = _R * (1 - 8 / 170)
DOUBLE_OUTER_RADIUS = _R

for k, v in _CAL_POINTS.items():
    _CAL_POINTS[k] = np.array(  # type: ignore
        [
            IMAGE_SIZE[0] / 2 + (v[0] * np.cos(v[1])),
            IMAGE_SIZE[1] / 2 - (v[0] * np.sin(v[1])),
        ]
    )


def display_preview(
    *images: Iterable[cv2.typing.MatLike | np.typing.NDArray[np.float32]],
    title: str = "Preview",
):
    tmp = map(
        lambda img: img
        if img.dtype == np.uint8  # type: ignore
        else np.array(img * 255, dtype=np.uint8),  # type: ignore
        images,
    )  # type: ignore
    tmp = np.hstack(list(tmp))  # type: ignore
    if tmp.dtype == np.float32:
        tmp = np.array(tmp * 255, dtype=np.uint8)

    cv2.imshow(title, tmp)
    k = cv2.waitKey(0)
    try:
        cv2.destroyWindow(title)
    except cv2.error:
        pass

    return k


def get_calibration_points() -> Dict[str, np.ndarray]:
    return _CAL_POINTS


def load_video(file: Path):
    video = cv2.VideoCapture(str(file))

    if not video.isOpened():
        raise IOError(f"Cannot open video: {file.as_posix()}")

    return video


def iter_video(video: cv2.VideoCapture):
    while True:  # Or while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        yield frame


def center_crop(image: np.ndarray, dsize: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    h_in, w_in = image.shape[:2]
    h_out, w_out = dsize

    if h_in == 0 or w_in == 0:
        return np.zeros((h_out, w_out, image.shape[2]), dtype=image.dtype)

    ratio_w = w_out / w_in
    ratio_h = h_out / h_in

    scale = 1.0
    if ratio_w > 1.0 or ratio_h > 1.0:
        scale = ratio_w if ratio_w > ratio_h else ratio_h

        new_w = int(w_in * scale)
        new_h = int(h_in * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h_in, w_in = new_h, new_w

    start_h = (h_in - h_out) // 2
    end_h = start_h + h_out
    start_w = (w_in - w_out) // 2
    end_w = start_w + w_out
    cropped_image = image[start_h:end_h, start_w:end_w]

    return cropped_image


def preprocess_frame(
    frame: np.ndarray, kernel_size: Tuple[int, int] = (3, 3), sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    if frame.dtype != np.uint8:
        raise TypeError(f"Frame is of type: {frame.dtype}, it should be: {np.uint8}")

    if frame.shape[-1] != 3:
        raise TypeError("Frame expected to be in BGR format (opencv).")

    result_color = cv2.GaussianBlur(frame, kernel_size, sigma)
    result_gray = cv2.cvtColor(result_color, cv2.COLOR_BGR2GRAY)
    result_gray = cv2.normalize(result_gray, result_gray, 255, 0, cv2.NORM_MINMAX)

    return (result_color, result_gray)


def load_model(path: Path) -> YOLO:
    try:
        model = YOLO(path)
    except NotImplementedError:
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        model = YOLO(path)
        pathlib.PosixPath = temp

    return model


def infer(model: YOLO, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    result = model.predict(image, verbose=False)[0]
    if result.boxes is None:
        return None

    if isinstance(result.boxes.cls, torch.Tensor):
        classes = result.boxes.cls.detach().cpu().numpy()
    else:
        classes = result.boxes.cls

    if isinstance(result.boxes.xywh, torch.Tensor):
        xywh = result.boxes.xywh.detach().cpu().numpy()
    else:
        xywh = result.boxes.xywh

    return classes, xywh


def fit_line(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    points = np.column_stack((x, y)).astype(np.float32)
    [vx], [vy], x0, y0 = cv2.fitLine(
        points, distType=cv2.DIST_FAIR, param=0, reps=0.01, aeps=0.01
    )
    k = float(vy / vx)
    d = float(y0[0] - k * x0[0])

    yfit = k * x + d
    residuals = y - yfit

    return yfit, residuals, k, d


def score_dart(p: np.ndarray):
    dx = p[0] - IMAGE_SIZE[0] / 2
    dy = IMAGE_SIZE[1] / 2 - p[1]

    radius = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx))

    if radius > DOUBLE_OUTER_RADIUS:
        return 0  # Missed the board

    multiplier = 1
    if radius <= INNER_BULL_RADIUS:
        return 50
    elif radius <= OUTER_BULL_RADIUS:
        return 25
    elif radius >= DOUBLE_INNER_RADIUS:
        multiplier = 2
    elif radius >= TRIPLE_INNER_RADIUS and radius <= TRIPLE_OUTER_RADIUS:
        multiplier = 3

    segments = np.array(
        [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5],
    )
    segments = segments[::-1]
    adjusted_angle = angle - 90

    segment_index = int((adjusted_angle - 9 + 360) % 360 // 18)
    base_score = segments[segment_index]

    return base_score * multiplier


def setup_camera(index: int = 1, backend: int = cv2.CAP_DSHOW) -> cv2.VideoCapture:
    cam = cv2.VideoCapture(index, backend)

    if not cam.isOpened():
        raise IOError(f"Cannot open camera with index {index}")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    w_default = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_default = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w_default == 0 or h_default == 0:
        raise IOError("Could not determine camera resolution/aspect ratio.")
    elif (w_default, h_default) == IMAGE_SIZE:
        return cam

    gcd = np.gcd(w_default, h_default)
    w_mul = w_default // gcd
    h_mul = h_default // gcd

    k_factor = int(max(np.ceil(IMAGE_SIZE[0] / w_mul), np.ceil(IMAGE_SIZE[1] / h_mul)))
    soll_w = k_factor * w_mul
    soll_h = k_factor * h_mul

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, soll_w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, soll_h)

    w_final = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_final = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w_final < IMAGE_SIZE[0]) or (h_final < IMAGE_SIZE[1]):
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        w_final = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_final = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cam

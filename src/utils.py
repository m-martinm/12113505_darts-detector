import logging
from pathlib import Path
from typing import Callable, Iterable, Tuple
import cv2
import numpy as np

IMAGE_SIZE = (640, 640)
gamma = 0.7
invGamma = 1.0 / gamma
lut_contrast = np.array(
    [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)], dtype=np.uint8
)
lut_lift_shadows = np.sqrt(np.arange(256, dtype=np.float32) * 255).astype(np.uint8)


def cv2numpy(img: cv2.typing.MatLike) -> np.typing.NDArray[np.float32]:
    return np.array(img / 255.0, dtype=np.float32)


def numpy2cv(img: np.typing.NDArray[np.float32]) -> np.typing.NDArray[np.uint8]:
    return np.array(img * 255, dtype=np.uint8)


def display_preview(
    *images: Iterable[cv2.typing.MatLike | np.typing.NDArray[np.float32]],
    title: str = "Preview",
):
    tmp = map(lambda img: img if img.dtype == np.uint8 else numpy2cv(img), images)
    tmp = np.hstack(list(tmp))
    if tmp.dtype == np.float32:
        tmp = numpy2cv(tmp)  # type: ignore

    cv2.imshow(title, tmp)
    k = cv2.waitKey(0)
    try:
        cv2.destroyWindow(title)
    except cv2.error:
        pass
    return k


def setup_camera(
    index: int = 1, backend: int = cv2.CAP_DSHOW, file: Path | None = None
) -> Tuple[cv2.VideoCapture, Callable[[np.ndarray], np.ndarray]]:
    
    if file is None:
        cam = cv2.VideoCapture(index, backend)
    else:
        cam = cv2.VideoCapture(str(file))
    
    if not cam.isOpened():
        raise IOError(f"Cannot open camera with index {index}")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    w_default = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_default = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w_default == 0 or h_default == 0:
        raise IOError("Could not determine camera resolution/aspect ratio.")
    elif (w_default, h_default) == IMAGE_SIZE:
        return cam, lambda x: x

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

    logging.info(f"Camera resolution set to {w_final}x{h_final}")

    def center_crop_func(frame: np.ndarray) -> np.ndarray:
        h_curr, w_curr = frame.shape[:2]
        h_target, w_target = IMAGE_SIZE
        if w_curr < w_target or h_curr < h_target:
            # logging.warning(
            #     f"Frame {w_curr}x{h_curr} is too small for target {w_target}x{h_target}. Returning original frame."
            # )
            return frame
        start_x = (w_curr - w_target) // 2
        start_y = (h_curr - h_target) // 2
        end_x = start_x + w_target
        end_y = start_y + h_target
        return frame[start_y:end_y, start_x:end_x]

    return cam, center_crop_func


def preprocess_frame(frame: np.ndarray):
    """Does gamma correction, small blur and normalization. INPLACE"""
    cv2.LUT(frame, lut_lift_shadows, frame)
    cv2.GaussianBlur(frame, (3, 3), 2, frame)
    cv2.normalize(frame, frame, 255, 0, cv2.NORM_MINMAX)

    return frame


def calculate_canny_diff(
    bg: np.ndarray,
    current: np.ndarray,
    lut: np.ndarray,
    th_low: float = 100.0,
    th_high: float = 200.0,
) -> np.ndarray:
    diff = cv2.absdiff(current, bg)
    diff = cv2.LUT(diff, lut)
    cv2.normalize(diff, diff, 255, 0, cv2.NORM_MINMAX)

    diff[diff < np.quantile(diff, 0.99)] = 0
    cv2.dilate(diff, np.ones((3, 3)), diff)
    canny_diff = cv2.Canny(diff, th_low, th_high)
    return canny_diff


def segment_dart(bg: np.ndarray, current: np.ndarray):
    diff = cv2.absdiff(bg, current)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    # cv2.normalize(diff, diff, 255, 0, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    total_change = cv2.countNonZero(thresh)
    if total_change == 0:
        return 0.0, thresh
    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(thresh, kernel, iterations=1)
    surviving_change = cv2.countNonZero(eroded_img)
    solidity_ratio = surviving_change / total_change

    return solidity_ratio, thresh


def get_frame(cam: cv2.VideoCapture, center_crop: Callable[[np.ndarray], np.ndarray]):
    ret, frame = cam.read()
    if not ret:
        raise IOError("Cannot read frame...")
    # curr_color = center_crop(frame)
    curr_color = cv2.resize(frame, IMAGE_SIZE)
    curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)
    curr_gray = preprocess_frame(curr_gray)

    return curr_gray, curr_color


def fitline_on_dart(x: np.ndarray, y: np.ndarray):
    points = np.column_stack((x, y)).astype(np.float32)
    [vx], [vy], x0, y0 = cv2.fitLine(
        points, distType=cv2.DIST_FAIR, param=0.2, reps=0.01, aeps=0.01
    )
    k = float(vy / vx)
    d = float(y0[0] - k * x0[0])

    yfit = k * x + d
    return yfit


def load_model(path: str):
    pass


def infer_model(model, input_data: np.ndarray):
    pass


if __name__ == "__main__":
    cam, center_crop = setup_camera()
    ret, frame = cam.read()
    frame = center_crop(frame)
    cv2.imshow("test", frame)
    cv2.displayOverlay("test", f"{frame.shape} shape")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cam.release()

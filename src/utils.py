import math
from typing import Callable, Iterable, Tuple
import cv2
import numpy as np

IMAGE_SIZE = (640, 640)


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
    cv2.waitKey(0)
    try:
        cv2.destroyWindow(title)
    except cv2.error:
        pass


def setup_camera(
    index: int = 1, backend: int = cv2.CAP_DSHOW
) -> Tuple[cv2.VideoCapture, Callable[[np.ndarray], np.ndarray]]:
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

    def center_crop_func(frame: np.ndarray) -> np.ndarray:
        h_curr, w_curr = frame.shape[:2]
        h_target, w_target = IMAGE_SIZE
        if w_curr < w_target or h_curr < h_target:
            print(
                f"Warning: Frame {w_curr}x{h_curr} is too small for target {w_target}x{h_target}. Returning original frame."
            )
            return frame
        start_x = (w_curr - w_target) // 2
        start_y = (h_curr - h_target) // 2
        end_x = start_x + w_target
        end_y = start_y + h_target
        return frame[start_y:end_y, start_x:end_x]

    return cam, center_crop_func


if __name__ == "__main__":
    cam, center_crop = setup_camera()
    ret, frame = cam.read()
    frame = center_crop(frame)
    cv2.imshow("test", frame)
    cv2.displayOverlay("test", f"{frame.shape} shape")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cam.release()

from typing import Iterable
import cv2
import numpy as np


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

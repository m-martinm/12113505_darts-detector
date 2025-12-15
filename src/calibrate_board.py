import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import utils

cal_points_src = {"D20_1": (), "D6_10": (), "D19_3": (), "D11_14": ()}

calibration_keys = list(cal_points_src.keys())
current_calibration = 0
calimage = None
calimage_show = None

WNAME = "Calibration"


def capture() -> np.ndarray:
    cam = utils.setup_camera()

    while True:
        ret, frame = cam.read()
        frame = utils.center_crop(frame)
        cv2.imshow("Capture", frame)
        k = cv2.waitKey(16)
        if k & 0xFF == 27:
            exit(0)
        elif k & 0xFF in (32, 10, 13):
            return frame


def on_mouse_cb(event, x, y, flags, param):
    global cal_points_src, current_calibration, calimage_show
    if (
        event == cv2.EVENT_LBUTTONDOWN
        and current_calibration < len(cal_points_src)
        and calimage_show is not None
        and calimage is not None
    ):
        cal_points_src[calibration_keys[current_calibration]] = np.array([x, y])  # type: ignore
        cv2.drawMarker(calimage_show, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 16)
        cv2.imshow(WNAME, calimage_show)
        current_calibration += 1
        if current_calibration < len(cal_points_src):
            cv2.displayOverlay(
                WNAME, f"Select the {calibration_keys[current_calibration]} point"
            )


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", type=str)
    parser.add_argument("--output", "-o", required=True, type=Path)

    args = parser.parse_args()
    logging.info(f"{args}")
    if not args.file:
        logging.info("No image provided, starting webcam.")
        args.rmbg = True
        calimage = capture()
    else:
        calimage = cv2.imread(args.file)
        if calimage is not None:
            logging.info(f"{args.file} loaded.")

    if calimage is None:
        exit(1)

    calimage_show = calimage.copy()

    cv2.imshow(WNAME, calimage_show)
    cv2.displayOverlay(
        WNAME, f"Select the {calibration_keys[current_calibration]} point"
    )
    cv2.setMouseCallback(WNAME, on_mouse_cb)

    while 1:
        k = cv2.waitKey(10)
        if k & 0xFF in (27, 113):
            exit(0)
        elif current_calibration == len(cal_points_src):
            break

    padding = 100  # in px
    r = min(utils.IMAGE_SIZE) / 2 - padding
    cal_points_dst = {
        "D20_1": (r, np.deg2rad(90 - 9)),
        "D6_10": (r, np.deg2rad(360 - 9)),
        "D19_3": (r, np.deg2rad(270 - 9)),
        "D11_14": (r, np.deg2rad(180 - 9)),
    }
    for k, v in cal_points_dst.items():
        cal_points_dst[k] = np.array(  # type: ignore
            [
                utils.IMAGE_SIZE[0] / 2 + int(v[0] * np.cos(v[1])),
                utils.IMAGE_SIZE[1] / 2 - int(v[0] * np.sin(v[1])),
            ]
        )

    H = cv2.findHomography(
        np.array(list(cal_points_src.values())),
        np.array(list(cal_points_dst.values())),
        cv2.RANSAC,
        10.0,
        confidence=0.99,
    )[0]

    calimage_show = cv2.warpPerspective(calimage_show, H, utils.IMAGE_SIZE)
    calimage = cv2.warpPerspective(calimage, H, utils.IMAGE_SIZE)
    cx = int(utils.IMAGE_SIZE[0] / 2)
    cy = int(utils.IMAGE_SIZE[1] / 2)
    r = int(r)
    Y, X = np.ogrid[: utils.IMAGE_SIZE[1], : utils.IMAGE_SIZE[0]]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    calimage_show = cv2.cvtColor(calimage_show, cv2.COLOR_BGR2BGRA)
    calimage = cv2.cvtColor(calimage, cv2.COLOR_BGR2BGRA)

    # calimage_show[dist_from_center > r] = (0, 0, 0, 0)
    # calimage[dist_from_center > r] = (0, 0, 0, 0)

    utils.display_preview(calimage)
    cv2.imwrite(str(args.output.with_suffix(".png")), calimage)

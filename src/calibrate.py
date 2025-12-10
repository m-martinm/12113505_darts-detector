import numpy as np
import cv2
from pathlib import Path

import utils

IMAGE_SIZE = (640, 640)

if __name__ == "__main__":
    # cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cam, crop = utils.setup_camera()

    # cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    # cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world spaceö
    imgpoints = []  # 2d points in image plane.

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        cv2.imshow("Webcam", frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = crop(frame_gray)
        ret, corners = cv2.findChessboardCorners(frame_gray, (7, 6), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                frame_gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(frame_gray, (7, 6), corners2, ret)
            cv2.imshow("Cal", frame_gray)

        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            exit(0)

        if len(imgpoints) > 10:
            print("Enough points...")
            break

    cam.release()

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        IMAGE_SIZE,
        None, # type: ignore
        None,  # type: ignore
    )
    print("ret: ", ret)
    print("mtx: ", mtx)
    print("dist: ", dist)
    print("rvecs: ", rvecs)
    print("tvecs: ", tvecs)

    cal_dir = Path(__file__).parent.parent / "data/calibration"
    # cal_file = cal_dir / ""
    np.savetxt(cal_dir / "mtx.txt", mtx)
    np.savetxt(cal_dir / "dist.txt", dist)

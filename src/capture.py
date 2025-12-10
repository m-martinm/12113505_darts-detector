import numpy as np
import cv2
from pathlib import Path

import utils

IMAGE_SIZE = (640, 640)

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    seq_dir = root / "data/sequence"
    cal_dir = root / "data/calibration"
    dist = np.loadtxt(cal_dir / "dist.txt")
    mtx = np.loadtxt(cal_dir / "mtx.txt")
    
    last_seq = max([int(d.name[-1]) for d in seq_dir.iterdir() if d.is_dir()])
    seq_idx = last_seq + 1
    outdir = seq_dir / f"seq{seq_idx}"
    outdir.mkdir()
    img_idx = 0

    cam, center_crop = utils.setup_camera()
    # cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        
        frame = center_crop(frame)
        
        # frame =cv2.undistort(frame, mtx, dist)
        
        cv2.imshow("Webcam", frame)
        k = cv2.waitKey(10)
        if k % 256 == 27:
            break
        elif k % 256 == 32:
            img_name = str(outdir / f"img{img_idx:02d}.jpg")
            cv2.imwrite(img_name, frame)
            cv2.displayOverlay(
                "Webcam", f"Image save to: {str(outdir / f'img{img_idx:02d}.jpg')}", 1000
            )
            img_idx += 1

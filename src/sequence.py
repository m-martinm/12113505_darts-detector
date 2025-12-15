from pathlib import Path
import time
from typing import Callable, List
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
import logging
from collections import deque

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.DEBUG
)
WARP_SIZE = utils.IMAGE_SIZE


def imgload(path: Path):
    img_color = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img_color is None:
        raise IOError("Couldn't load board image." + str(path))

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGRA2GRAY)

    return (img_color, img_gray)


def do_sift(board_img: cv2.typing.MatLike, scene_img: cv2.typing.MatLike):
    sift = cv2.SIFT.create()

    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_img, None)  # type: ignore
    board_keypoints, board_descriptors = sift.detectAndCompute(board_img, None)  # type: ignore

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore

    matches: List[cv2.DMatch] = [
        m
        for m, n in flann.knnMatch(scene_descriptors, board_descriptors, k=2)
        if m.distance < 0.8 * n.distance
    ]

    board_points = np.array(
        [board_keypoints[m.trainIdx].pt for m in matches], dtype=np.float32
    ).reshape(-1, 1, 2)
    scene_points = np.array(
        [scene_keypoints[m.queryIdx].pt for m in matches], dtype=np.float32
    ).reshape(-1, 1, 2)

    H_board = cv2.findHomography(
        board_points, scene_points, cv2.RANSAC, 10.0, confidence=0.99
    )[0]
    H_scene = cv2.findHomography(
        scene_points, board_points, cv2.RANSAC, 10.0, confidence=0.99
    )[0]
    return (H_board, H_scene)



def load_board_reference(path: Path):
    board_color, board_gray = imgload(path)
    board_mask = board_color[:, :, 3] != 0

    board_mask = cv2.GaussianBlur(
        cv2.dilate(board_mask.astype(np.uint8) * 255, np.ones((3, 3))), (3, 3), 1
    ).clip(0, 128)
    cv2.threshold(board_mask, 0, 255, cv2.THRESH_BINARY, board_mask)
    board_color[~board_mask.astype(np.bool)] = (0, 0, 0, 0)
    board_gray[~board_mask.astype(np.bool)] = 0
    cv2.LUT(board_gray, utils.lut_lift_shadows, board_gray)

    return board_color, board_gray, board_mask




if __name__ == "__main__":
    root = Path(__file__).parent.parent.resolve()

    board_path = root / r"data\sift\bg_removed\board3.png"
    board_color, board_gray, board_mask = load_board_reference(board_path)
    
    # cam, center_crop = utils.setup_camera(0, cv2.CAP_DSHOW)
    cam, center_crop = utils.setup_camera(file=Path(r"G:\My Drive\IMG_1320.mov"))
    cv2.namedWindow("Dart")
    cv2.displayOverlay(
        "Dart", "Position the camera and fix it\nPRESS SPACE TO CONTINUE"
    )
    while 1:
        curr_gray, curr_color = utils.get_frame(cam, center_crop)
        cv2.imshow("Dart", curr_color)
        k = cv2.waitKey(30)
        if k & 0xFF == 32:
            break
        elif k & 0xFF in (27, 113):
            exit(0)

    cv2.displayOverlay(
        "Dart", "Position the camera and fix it\nPRESS SPACE TO CONTINOUE", 10
    )
    bg_gray, bg_color = utils.get_frame(cam, center_crop)
    H_board, H_scene = do_sift(board_gray, bg_gray)
    cam_is_left = H_scene[0, -1] > 0

    bg_color = cv2.warpPerspective(bg_color, H_scene, WARP_SIZE)
    bg_gray = cv2.warpPerspective(bg_gray, H_scene, WARP_SIZE)
    bg_copy = bg_color.copy()

    buffer_length = 5
    buffer_cursor = 0
    buffer = np.empty(shape=(buffer_length, 4), dtype=np.float32)
    pt1 = (0, 0)
    pt2 = (0, 0)
    # sliding_window_size = 30
    # sliding_window = deque([200.0 for _ in range(sliding_window_size)], maxlen=sliding_window_size)
    
    while 1:
        time_st = time.time_ns()
        curr_gray, curr_color = utils.get_frame(cam, center_crop)
        curr_color = cv2.warpPerspective(curr_color, H_scene, WARP_SIZE)
        curr_gray = cv2.warpPerspective(curr_gray, H_scene, WARP_SIZE)
        ratio, dart = utils.segment_dart(bg_gray, curr_gray)
        # cv2.imshow("Segmentation", dart)
        # current_mean_y_std = np.mean(np.array(sliding_window))
        # current_std_y_std = np.std(np.array(sliding_window))
        # y_std = np.argwhere(dart)[:,0].std().mean()
        # logging.debug( f"{y_std:.3f}, {current_mean_y_std:.3f}, {current_std_y_std:.3f}")
        
        
        # if current_mean_y_std - 3 * current_std_y_std > y_std:
        if ratio > 0.5:
            cv2.imshow("Segmentation", dart)
            cv2.displayOverlay("Segmentation", f"Ratio: {ratio:.3f}")
            # cv2.displayOverlay("Segmentation", f"{dart.mean()=}")
            logging.debug("Maybe dart")
            y, x = np.where(dart)
            yfit = utils.fitline_on_dart(x, y)

            residuals = y - yfit
            sigma_residuals = np.std(residuals)
            outlier_mask = np.abs(residuals) > sigma_residuals * 2

            if outlier_mask.mean() < 0.2:  # 0.1 was
                y = y[~outlier_mask]
                x = x[~outlier_mask]
                yfit = utils.fitline_on_dart(x, y)

                residuals = y - yfit
                sigma_residuals = np.std(residuals)
                outlier_mask = np.abs(residuals) > sigma_residuals * 2

                if outlier_mask.mean() < 0.1:
                    xmin = x[~outlier_mask].min()
                    xmax = x[~outlier_mask].max()
                    buffer[buffer_cursor] = np.array(
                        [[xmin, yfit[-1], xmax, yfit[0]]], dtype=np.float32
                    )
                    # frame_buffer[:,320*buffer_cursor:320*buffer_cursor + 320] = cv2.resize(curr_gray, (320, 320))
                    buffer_cursor += 1
                    logging.debug(f"Dart found {buffer_cursor=}")
                else:
                    buffer_cursor = 0
            else:
                logging.debug("Too many outliers, resetting buffer")
                buffer_cursor = 0

        else:
            buffer_cursor = 0
            # sliding_window.append(y_std)

        if buffer_cursor >= buffer_length - 1:
            logging.debug("Buffer full, resetting buffer and checking it")
            # cv2.imshow("Frame buffer", frame_buffer[:, 3*320:8*320])
            buffer_cursor = 0
            tmp = buffer[1:-1]  # discard last 2 and first 3

            mean_of_std = np.mean(np.std(tmp, axis=0))
            logging.debug(f"{mean_of_std=}")
            if mean_of_std < 5.0:
                pt1_x = int(tmp[:, 0].mean())
                pt1_y = int(tmp[:, 1].mean())
                pt2_x = int(tmp[:, 2].mean())
                pt2_y = int(tmp[:, 3].mean())

                pt1 = (pt1_x, pt1_y)
                pt2 = (pt2_x, pt2_y)

        if buffer_cursor == 0:
            cv2.line(curr_color, pt1, pt2, (0, 0, 255), 1)
            cv2.imshow("Dart", curr_color)
            k = cv2.waitKey(10)
            if k % 0xFF == 27:
                break
            elif k % 0xFF == 32:
                tmp, _ = utils.get_frame(cam, center_crop)
                H_board, H_scene = do_sift(board_gray, tmp)
                cam_is_left = H_scene[0, -1] > 0

            bg_color = curr_color
            bg_gray = curr_gray

        # frame_time = int((time.time_ns() - time_st) * 1e-6)
        # if (dt := 40 - frame_time) > 0:
        #     k = cv2.waitKey(dt)
        #     if k % 0xFF == 27:
        #         break
        #     elif k % 0xFF == 32:
        #         tmp, _ = utils.get_frame(cam, center_crop)
        #         H_board, H_scene = do_sift(board_gray, tmp)
        #         cam_is_left = H_scene[0, -1] > 0

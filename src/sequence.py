from pathlib import Path
from typing import List
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt


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


def calculate_luts(gamma: float = 0.7):
    invGamma = 1.0 / gamma
    lut_contrast = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)], dtype=np.uint8
    )

    lut_lift_shadows = np.sqrt(np.arange(256, dtype=np.float32) * 255).astype(np.uint8)
    return lut_contrast, lut_lift_shadows


def load_board_reference(path: Path):
    board_color, board_gray = imgload(board_path)
    board_mask = board_color[:, :, 3] != 0

    board_mask = cv2.GaussianBlur(
        cv2.dilate(board_mask.astype(np.uint8) * 255, np.ones((3, 3))), (3, 3), 1
    ).clip(0, 128)
    cv2.threshold(board_mask, 0, 255, cv2.THRESH_BINARY, board_mask)
    board_color[~board_mask.astype(np.bool)] = (0, 0, 0, 0)
    board_gray[~board_mask.astype(np.bool)] = 0
    cv2.LUT(board_gray, lut_lift_shadows, board_gray)

    return board_color, board_gray, board_mask


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
    cv2.normalize(diff, diff, 255, 0, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # utils.display_preview(thresh)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5)))

    return thresh

def get_frame(cam: cv2.VideoCapture):
    ret, frame = cam.read()
    if not ret:
        raise IOError("Coudln't read frame...")
    pass

if __name__ == "__main__":
    lut_contrast, lut_lift_shadows = calculate_luts()

    root = Path(__file__).parent.parent.resolve()
    # cal_dir = root / "data/calibration"
    # dist = np.loadtxt(cal_dir / "dist.txt")
    # mtx = np.loadtxt(cal_dir / "mtx.txt")

    board_path = root / r"data\sift\bg_removed\board1.png"
    board_color, board_gray, board_mask = load_board_reference(board_path)

    # sequence_dir = root / "data/sequence/seq3"
    # sequence = [imgload(k) for k in sequence_dir.glob("*.jpg")]

    # bg_color, bg_gray = sequence.pop(0)
    # preprocess_frame(bg_gray)
    cam, center_crop = utils.setup_camera()
    
    cv2.namedWindow("Dart")
    cv2.displayOverlay("Dart", "PRESS SPACE TO START")
    while 1:
        try:
            ret, frame = cam.read()
            if not ret:
                break
            curr_color = center_crop(frame)

        except IndexError:
            break
        cv2.imshow("Dart", curr_color)
        k = cv2.waitKey(10)
        # print(ord(" "), ord("q"))
        if k & 0xFF == 32:
            break
        elif k & 0xFF in (27, 113):
            exit(0)
    cv2.displayOverlay("Dart", "a", 10)
    # print(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cam.read()
    bg_color = center_crop(frame)
    bg_gray = cv2.cvtColor(bg_color, cv2.COLOR_BGR2GRAY)
    bg_gray = preprocess_frame(bg_gray)
    
    H_board, H_scene = do_sift(board_gray, bg_gray)
    # print(H_board)
    # print(np.linalg.inv(H_board))
    # print(H_scene)
    H_inv = np.linalg.inv(H_board)
    cam_is_left = H_scene[0, -1] < 0
    cam_is_down = H_scene[1, -1] > 0

    h, w = board_gray.shape
    board_corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
    ).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(board_corners, H_board).astype(np.int32)

    tmp = scene_corners[1,0,:] - scene_corners[0,0,:]
    board_size_pixels = np.sqrt(tmp.dot(tmp))
    pt1 = (0,0)
    pt2 = (0,0)
    # print(board_size_pixels)
    # exit(0)
    
    while 1:
        try:
            ret, frame = cam.read()
            if not ret:
                break
            curr_color = center_crop(frame)
            curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)
            # curr_color, curr_gray = sequence.pop(0)
            preprocess_frame(curr_gray)
        except IndexError:
            break
        
        
        dart = segment_dart(bg_gray, curr_gray)
        if dart.mean() < 10:
            cv2.waitKey(10)
            ret, frame = cam.read()
            if not ret:
                break
            curr_color = center_crop(frame)
            curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)
            preprocess_frame(curr_gray)

            y, x = np.where(dart)
            points = np.column_stack((x, y)).astype(np.float32)
            [vx], [vy], x0, y0 = cv2.fitLine(
                points, distType=cv2.DIST_FAIR, param=0.2, reps=0.01, aeps=0.01
            )
            k = float(vy / vx)
            d = float(y0[0] - k * x0[0])

            yfit = k * x + d
            residuals = y - yfit
            sigma_residuals = np.std(residuals)
            outlier_mask = np.abs(residuals) > sigma_residuals * 2
            if outlier_mask.mean() < 0.1:
                xmin = int(x[~outlier_mask].min())
                xmax = int(x[~outlier_mask].max())

                pt1 = np.array((xmin, int(k * xmin + d)), dtype=np.int32)
                pt2 = np.array((xmax, int(k * xmax + d)), dtype=np.int32)
                length = pt2- pt1
                length = np.sqrt(length.dot(length))
                # print(length, board_size_pixels)
                if length * 2 < board_size_pixels:
                    cv2.line(curr_color, pt1, pt2, (0, 0, 255), 1)
                    cv2.polylines(curr_color, [scene_corners], True, (0, 255, 255), 2)
                
            else:
                cv2.displayOverlay("Dart", "New round, no dart hit", 3000)
                cv2.waitKey(3000)
        else:
            cv2.line(curr_color, pt1, pt2, (0, 0, 255), 1)
            cv2.polylines(curr_color, [scene_corners], True, (0, 255, 255), 2)

        
        cv2.imshow("Dart", curr_color)
        k = cv2.waitKey(10)
        if k % 0xFF == 27:
            break
        
        bg_color = curr_color
        bg_gray = curr_gray

        # utils.display_preview(curr_color)
        # if update_bg:
        #     bg_color = curr_color
        #     bg_gray = curr_gray
            
        # k = cv2.waitKey(5)
        # if k & 256 == 27:
        #     break


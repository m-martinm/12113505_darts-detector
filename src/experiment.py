from collections import deque
import logging
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sequence import load_board_reference, do_sift
import utils
from sklearn.cluster import DBSCAN, dbscan

rng = np.random.default_rng(6)
DATASET_SIZE = 800


def add_gauss_noise(
    img: np.ndarray, mean: float = 0.0, sigma: float = 0.1
) -> np.ndarray:
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img + gauss
    return noisy.astype(img.dtype)


def add_salt_and_pepper_noise(
    img: np.ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5
) -> np.ndarray:
    noisy = np.copy(img)
    num_salt = np.ceil(amount * img.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[coords[0], coords[1]] = 1

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy.astype(img.dtype)


def add_speckle_noise(
    img: np.ndarray, mean: float = 0.0, sigma: float = 0.1
) -> np.ndarray:
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img + img * gauss
    return noisy.astype(img.dtype)


def add_poisson_noise(img: np.ndarray) -> np.ndarray:
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    return noisy.astype(img.dtype)


def prepare_image(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = add_speckle_noise(img, rng.uniform(1e-6, 1e-4), rng.uniform(1e-2, 1e-1))
    img = add_salt_and_pepper_noise(img, rng.uniform(1e-5, 1e-4), rng.uniform(0.3, 0.7))
    img = utils.preprocess_frame(img)
    return img


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    # DEBUG = "debug" in sys.argv
    board = load_board_reference(Path("data/sift/bg_removed/board4.png"))[1]

    root = Path(__file__).parent.parent
    # old_dataset_dir = root / "data" / "darts_positions_old"
    old_dataset_dir = Path(r"G:\My Drive\YOLOTraining\test")
    new_dataset_dir = root / "data" / "darts_positions"
    images_input = sorted(
        old_dataset_dir.rglob("*.jpg"), key=lambda p: int(p.stem[-5:].removeprefix("_"))
    )
    labels_input = sorted(
        old_dataset_dir.rglob("*.txt"), key=lambda p: int(p.stem[-5:].removeprefix("_"))
    )
    logging.info(f"Found {len(images_input)} images in {old_dataset_dir}")
    logging.info(f"Found {len(labels_input)} labels in {old_dataset_dir}")

    sliding_window_size = 30
    sliding_window = deque(
        [200.0 for _ in range(sliding_window_size)], maxlen=sliding_window_size
    )
    image_idx = 0
    for idx, (img_path, label_path) in enumerate(zip(images_input, labels_input), 1):
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning(f"Could not read image {img_path}, skipping.")
            continue
        img = prepare_image(img)
        if idx % 4 == 0 or idx == 1:
            logging.info(f"Sequence {idx // 4 + 1}")
            H_board, H_scene = do_sift(board, img)
            bg_img = cv2.warpPerspective(img, H_scene, (DATASET_SIZE, DATASET_SIZE))
            continue

        img = cv2.warpPerspective(img, H_scene, (DATASET_SIZE, DATASET_SIZE))
        # dart = utils.segment_dart(bg_img, img)
        diff = cv2.absdiff(bg_img, img)
        # diff = cv2.GaussianBlur(diff, (3, 3), 0)
        # cv2.normalize(diff, diff, 255, 0, cv2.NORM_MINMAX)
        z = (diff - diff.mean()) / diff.std()
        # diff = np.uint8(np.clip((z * 16 + 128), 0, 255))
        diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        plt.imshow(diff, cmap='gray')
        plt.colorbar()
        plt.show()
        # print(diff.shape)
        # cv2.imshow("Diff", diff)
        # cv2.waitKey(0)
                
        
        # current_mean_y_std = np.mean(np.array(sliding_window))
        # current_std_y_std = np.std(np.array(sliding_window))
        # points = np.argwhere(dart)
        # y_std = points[:, 0].std()
        # if (
        #     current_mean_y_std - 2 * current_std_y_std < y_std
        #     or dart.mean() / 255 > 0.3
        # ):
        #     sliding_window.append(y_std)
        #     logging.info("Dropping frame due to high y_std")
        #     bg_img = img
        #     continue

        # src = np.loadtxt(label_path, dtype=np.float32)
        # dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H_scene).reshape(2)
        # if DEBUG:
        #     disp = cv2.drawMarker(
        #         cv2.cvtColor(dart, cv2.COLOR_GRAY2BGR),
        #         dst.astype(np.int32),  # type: ignore
        #         color=(0, 0, 255),
        #         markerType=cv2.MARKER_CROSS,
        #         markerSize=20,
        #         thickness=2,
        #         line_type=cv2.LINE_AA,
        #     )  # type: ignore
        #     dst /= DATASET_SIZE
        #     cv2.imshow("Processed Image", disp)
        #     cv2.displayOverlay("Processed Image", f"Source: {src}, Dest: {dst}")
        #     k = cv2.waitKey(0)
        #     if k & 0xFF == 27:
        #         break
        # else:
        #     dst /= DATASET_SIZE
        #     if dst.max() < 1 and dst.min() > 0:
        #         np.savetxt(new_dataset_dir / f"labels/img_{image_idx:05d}.txt", dst)
        #         cv2.imwrite(str(new_dataset_dir / f"images/img_{image_idx:05d}.jpg"), dart)
        #         image_idx += 1
        #     else:
        #         logging.warning("Dart position out of bounds, skipping frame.")
        bg_img = img

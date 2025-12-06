from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root = (Path(__file__).parent / "../../").resolve()

    scene_img_path = root / "data/real_images/IMG_1280.jpg"
    scene_img = cv2.imread(str(scene_img_path))
    if scene_img is None:
        raise FileNotFoundError("Couldn't load image " + str(scene_img_path))
    scene_img_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

    object_img_path = root / "data/sift/bg_removed/board1.jpg"
    object_img = cv2.imread(str(object_img_path))
    if object_img is None:
        raise FileNotFoundError("Couldn't load image " + str(object_img_path))
    object_img_gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(object_img_gray, None)
    kp2, des2 = sift.detectAndCompute(scene_img_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    MIN_MATCH_COUNT = 10
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = object_img_gray.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(scene_img_gray, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        exit(1)

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )

    img3 = cv2.drawMatches(object_img_gray, kp1, scene_img_gray, kp2, good, None, **draw_params)

    plt.imshow(img3, "gray")
    plt.show()

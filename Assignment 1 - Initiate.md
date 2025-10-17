# Dart Detection

## Topic

This is a "beat the classics" type of project, with the goal to compare traditional computer vision techniques with machine learning methods in the context of darts.

## Summary 

### Short description

In this project I want to test different approaches for detecting dart locations on a dartboard and automatically calculating the score. This can be done without machine learning if multiple camera angles are used. However, the key difference in this project is that it will use a single webcam. While traditional methods can still work with calibration, I plan to compare their performance with machine learning-based approaches.

Without using machine learning, I have the following ideas for detecting dart positions:
- **Background subtraction**: Take a background image of the empty dartboard, then take a new image after each dart throw. Subtract the background from the new image and isolate the dart. If the camera has been calibrated with a perspective transform matrix, I can filter the resulting difference image to find pixel blobs, apply the perspective transform, and map the coordinates onto the board.
    - I could also use Canny edge detection to improve the accuracy of the results.
    - The Hough circle transform might be also used to find the dartboard on the image.
- **SIFT-based detection**: As an alternative, I could use SIFT (Scale-Invariant Feature Transform) to find features of both the darts and the board, and then apply the same mapping to determine dart locations.

For the deep learning approach, I will need a labeled dataset. I plan to annotate around 500 images with bounding boxes. However, this data is likely not enough to train a model from scratch.

One idea is to use a pre-trained YOLO or R-CNN model and fine-tune it on my dataset. This model could then be used to label a larger dataset. The setup would allow more experimentation on the smaller model with different architectures.

Another approach is to take images of the same dart positions under different lighting conditions (direct light, diffused light etc.). This would effectively double or triple the dataset size without requiring additional manual labeling. I could further expand the dataset using data augmentation techniques such as rotations, perspective transforms during preprocessing. With these steps combined, I expect to reach a dataset size of around 2000 – 3000 images. While this is still relatively small, it should be enough for experimentation.

### Work-breakdown

| Work                                   | Approximation |
| -------------------------------------- | ------------- |
| Taking images                          | ~ 2hours      |
| Labeling                               | ~ 8 hours     |
| Non-machine learning approaches        | 10 - 15 hours |
| Machine learning approaches            | 20 - 25 hours |
| Building a Python program to use these | 10 - 15 hours |
| Final report                           | ~ 6 hours     |
| Presentation                           | ~ 4 hours     |
|                                        |               |
|                                        | 60 - 70 hours |

## References

William McNally et al, "DeepDarts: Modeling Keypoints as Objects for Automatic Scorekeeping in Darts using a Single Camera," 2021. 
https://arxiv.org/abs/2105.09880
https://ieee-dataport.org/open-access/deepdarts-dataset

Liu, Y.-C., Ma, C.-Y., He, Z., Kuo, C.-W., Chen, K., Zhang, P., Wu, B., Kira, Z., & Vajda, P. (2021). Unbiased Teacher for Semi-Supervised Object Detection. ICLR 2021.
https://ycliu93.github.io/projects/unbiasedteacher.html

J. Canny, "A Computational Approach to Edge Detection," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PAMI-8, no. 6, pp. 679-698, Nov. 1986, doi: 10.1109/TPAMI.1986.4767851.
https://ieeexplore.ieee.org/document/4767851

Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94

# Assignment 2 - Hacking

## Metric

In the [metrics](metrics.md) file I tried to come up with a good metric between the two approaches. At the end I choose the 3 to compare them.

1. The number of found dart hits on the board
2. The mean error in L2 distance of the hits
3. The frame processing time

I was hoping for a better "found" rate, especially with the YOLO model, but the avarage mean distance is actually better than I hoped for.

## Work-breakdown

Compared to the initial planning, I did not hand annotate the images. I used the Godot game enginge to generate the data so my work-breakdown changed quiet a bit.

### Original 

| Work                                   | Approximation |
| -------------------------------------- | ------------- |
| Taking images                          | ~ 2hours      |
| Labeling                               | ~ 8 hours     |
| Non-machine learning approaches        | 10 - 15 hours |
| Machine learning approaches            | 20 - 25 hours |
| Building a Python program to use these | 10 - 15 hours |

### Actual

| Work                                   | Approximation |
| -------------------------------------- | ------------- |
| Setting up simulation enviroment       | ~ 16 hours    |
| Labeling (Generating the images)       | ~ 1 hour      |
| Non-machine learning approaches        | ~ 14 hours    |
| Training the YOLO model                | ~ 6 hours     |
| Building a Python program to use these | ~ 16 hours    |

Plus I tried to train a CordCNN model, which took quiet a bit of time (~20 hours). But sadly at the end I couldn't use it, since it's performance was worse than I expected.




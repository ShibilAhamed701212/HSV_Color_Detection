"""
Run main.py against a synthetic scene instead of a webcam and save what it renders.

Used to produce the README screenshots and as a quick smoke test without a camera.
Needs the GUI build of OpenCV (opencv-python) and a display; on a headless Linux box
use `xvfb-run -a python scripts/synthetic_demo.py`.
"""

import os
import runpy
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "docs", "screenshots")
FRAMES = 5


def synthetic_scene():
    frame = np.full((480, 640, 3), (90, 90, 90), np.uint8)
    # Red disc whose two halves sit either side of the hue wrap point (~1 and ~177)
    cv2.circle(frame, (320, 240), 70, (0, 10, 220), -1)
    right_half = np.zeros((480, 640), np.uint8)
    cv2.circle(right_half, (320, 240), 70, 255, -1)
    right_half[:, :320] = 0
    frame[right_half > 0] = (25, 0, 220)
    cv2.circle(frame, (120, 380), 40, (0, 0, 200), -1)  # second red object
    cv2.rectangle(frame, (470, 300), (590, 420), (200, 80, 20), -1)  # blue distractor
    return frame


class SyntheticCapture:
    def __init__(self, *args):
        pass

    def isOpened(self):
        return True

    def read(self):
        return True, synthetic_scene()

    def release(self):
        pass


def main():
    shown = {}
    real_imshow, real_wait_key = cv2.imshow, cv2.waitKey
    frame_count = {"n": 0}

    def imshow(name, image):
        real_imshow(name, image)
        shown[name] = image.copy()

    def wait_key(delay):
        real_wait_key(1)
        frame_count["n"] += 1
        return ord("q") if frame_count["n"] >= FRAMES else -1

    cv2.VideoCapture = SyntheticCapture
    cv2.imshow = imshow
    cv2.waitKey = wait_key

    sys.path.insert(0, REPO)
    runpy.run_path(os.path.join(REPO, "main.py"), run_name="__main__")

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, "synthetic_detection.png"), shown["HSV Color Detection"])
    cv2.imwrite(os.path.join(OUT_DIR, "synthetic_mask.png"), shown["Binary Mask"])
    print(f"Saved screenshots to {OUT_DIR}")


if __name__ == "__main__":
    main()

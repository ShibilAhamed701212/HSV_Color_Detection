import cv2
import numpy as np


def get_limits(color, hue_delta=15, min_sat=40, min_val=40):
    """
    Given a BGR color [B, G, R], returns HSV lower and upper bounds.
    If the color hue wraps around boundary (e.g., Red around 0/180),
    returns ((lower1, upper1), (lower2, upper2)).
    Otherwise returns ((lower1, upper1), None).
    """
    c = np.array([[color]], dtype=np.uint8)
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = int(hsvC[0][0][0])
    sat = int(hsvC[0][0][1])
    val = int(hsvC[0][0][2])

    effective_sat = max(min_sat, max(20, sat - 60))
    effective_val = max(min_val, max(20, val - 60))

    lower_hue = hue - hue_delta
    upper_hue = hue + hue_delta

    if lower_hue < 0:
        range1 = (
            np.array([180 + lower_hue, effective_sat, effective_val], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        range2 = (
            np.array([0, effective_sat, effective_val], dtype=np.uint8),
            np.array([upper_hue, 255, 255], dtype=np.uint8),
        )
        return range1, range2
    elif upper_hue > 179:
        range1 = (
            np.array([lower_hue, effective_sat, effective_val], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        range2 = (
            np.array([0, effective_sat, effective_val], dtype=np.uint8),
            np.array([upper_hue - 180, 255, 255], dtype=np.uint8),
        )
        return range1, range2
    else:
        range1 = (
            np.array([lower_hue, effective_sat, effective_val], dtype=np.uint8),
            np.array([upper_hue, 255, 255], dtype=np.uint8),
        )
        return range1, None


def get_color_mask(hsv_frame, color, hue_delta=15, min_sat=40, min_val=40):
    """
    Generates a clean, noise-filtered binary mask for a target BGR color.
    Includes morphological opening and closing for maximum detection accuracy.
    """
    range1, range2 = get_limits(
        color, hue_delta=hue_delta, min_sat=min_sat, min_val=min_val
    )

    mask1 = cv2.inRange(hsv_frame, range1[0], range1[1])
    if range2 is not None:
        mask2 = cv2.inRange(hsv_frame, range2[0], range2[1])
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = mask1

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def clean_mask(mask, kernel_size=5):
    """
    Applies morphological opening and closing to remove noise specks and fill holes.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def merge_nearby_boxes(boxes, max_gap=40):
    """
    Merges bounding boxes (x, y, w, h) that are within max_gap pixels of each other.
    Useful when glasses, eyebrows, or shadows split a single entity into multiple fragments.
    """
    if not boxes:
        return []
    rects = [[x, y, x + w, y + h] for (x, y, w, h) in boxes]
    merged = True
    while merged:
        merged = False
        new_rects = []
        while rects:
            curr = rects.pop(0)
            has_merged = False
            for i in range(len(new_rects)):
                other = new_rects[i]
                if not (
                    curr[2] + max_gap < other[0]
                    or curr[0] - max_gap > other[2]
                    or curr[3] + max_gap < other[1]
                    or curr[1] - max_gap > other[3]
                ):
                    new_rects[i] = [
                        min(curr[0], other[0]),
                        min(curr[1], other[1]),
                        max(curr[2], other[2]),
                        max(curr[3], other[3]),
                    ]
                    has_merged = True
                    merged = True
                    break
            if not has_merged:
                new_rects.append(curr)
        rects = new_rects
    return [(r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in rects]


def circular_hue_median(hues):
    """
    Median of OpenCV hue values (0-179) treating hue as a circle, so a red patch
    with pixels at both 2 and 177 yields a red hue instead of ~89 (cyan).
    """
    hues = np.asarray(hues, dtype=np.float64)
    angles = hues * (2 * np.pi / 180.0)
    mean = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    mean_h = (mean * 180.0 / (2 * np.pi)) % 180.0
    # Signed distance of each hue from the circular mean, in [-90, 90)
    offsets = (hues - mean_h + 90.0) % 180.0 - 90.0
    return round(float(mean_h + np.median(offsets))) % 180


def calculate_smart_hsv_bounds(hsv_patch):
    """
    Analyzes an HSV pixel patch and intelligently determines bounds.
    Distinguishes White, Black, Gray, and Vibrant Color objects.
    """
    if len(hsv_patch) == 0:
        return (0, 179, 40, 255, 40, 255, "Default")

    median_h = int(np.median(hsv_patch[:, 0]))
    median_s = int(np.median(hsv_patch[:, 1]))
    median_v = int(np.median(hsv_patch[:, 2]))

    # Case 1: White / Bright Light Object (like earbud case)
    if median_s < 40 and median_v > 160:
        h_min, h_max = 0, 179
        s_min = 0
        s_max = max(50, median_s + 40)
        v_min = max(140, median_v - 60)
        v_max = 255
        label = f"White Object (S:{median_s} V:{median_v})"

    # Case 2: Dark / Black Object
    elif median_v < 60:
        h_min, h_max = 0, 179
        s_min = 0
        s_max = 255
        v_min = 0
        v_max = max(70, median_v + 35)
        label = f"Black Object (V:{median_v})"

    # Case 3: Grayscale / Neutral Gray Object
    elif median_s < 35 and 60 <= median_v <= 160:
        h_min, h_max = 0, 179
        s_min = 0
        s_max = max(45, median_s + 30)
        v_min = max(35, median_v - 45)
        v_max = min(230, median_v + 45)
        label = f"Gray Object (S:{median_s} V:{median_v})"

    # Case 4: Vibrant / Saturated Color Object (Red, Green, Blue, Yellow, Orange, etc.)
    else:
        # Hue is circular (0 and 179 are neighbours), so use a circular median and
        # let the range wrap. h_min > h_max means "wraps through 0"; main.py builds
        # a dual-range mask for that case.
        median_h = circular_hue_median(hsv_patch[:, 0])
        h_min = (median_h - 12) % 180
        h_max = (median_h + 12) % 180
        s_min = max(40, median_s - 50)
        s_max = min(255, median_s + 60)
        v_min = max(40, median_v - 50)
        v_max = min(255, median_v + 60)
        label = f"Color (H:{median_h} S:{median_s} V:{median_v})"

    return (h_min, h_max, s_min, s_max, v_min, v_max, label)


def get_bounds_from_sample(hsv_pixel, hue_delta=15, sat_delta=70, val_delta=70):
    """
    Given a sampled HSV pixel tuple (h, s, v), returns lower and upper HSV bounds.
    Handles red hue wraparound if necessary.
    """
    h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

    lower_h = h - hue_delta
    upper_h = h + hue_delta

    min_s = max(20, s - sat_delta)
    max_s = min(255, s + sat_delta)

    min_v = max(20, v - val_delta)
    max_v = min(255, v + val_delta)

    if lower_h < 0:
        range1 = (
            np.array([180 + lower_h, min_s, min_v], dtype=np.uint8),
            np.array([179, max_s, max_v], dtype=np.uint8),
        )
        range2 = (
            np.array([0, min_s, min_v], dtype=np.uint8),
            np.array([upper_h, max_s, max_v], dtype=np.uint8),
        )
        return range1, range2
    elif upper_h > 179:
        range1 = (
            np.array([lower_h, min_s, min_v], dtype=np.uint8),
            np.array([179, max_s, max_v], dtype=np.uint8),
        )
        range2 = (
            np.array([0, min_s, min_v], dtype=np.uint8),
            np.array([upper_h - 180, max_s, max_v], dtype=np.uint8),
        )
        return range1, range2
    else:
        range1 = (
            np.array([lower_h, min_s, min_v], dtype=np.uint8),
            np.array([upper_h, max_s, max_v], dtype=np.uint8),
        )
        return range1, None


if __name__ == "__main__":
    print("=== HSV Color Detection Utility Module Test ===")
    yellow_bgr = [0, 255, 255]
    range1, range2 = get_limits(yellow_bgr)
    print(f"Sample BGR Color (Yellow): {yellow_bgr}")
    print(f"  Lower HSV Bound: {range1[0]}")
    print(f"  Upper HSV Bound: {range1[1]}")
    if range2 is not None:
        print(f"  Wrapped Lower HSV Bound: {range2[0]}")
        print(f"  Wrapped Upper HSV Bound: {range2[1]}")
    print("===============================================")
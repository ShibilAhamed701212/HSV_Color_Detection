import cv2
import numpy as np
import pytest

from util import (
    calculate_smart_hsv_bounds,
    circular_hue_median,
    clean_mask,
    get_bounds_from_sample,
    get_limits,
    merge_nearby_boxes,
)


def hue_in_range(h, h_min, h_max):
    """Mirror of main.py's mask logic: h_min > h_max means the range wraps through 0."""
    if h_min <= h_max:
        return h_min <= h <= h_max
    return h >= h_min or h <= h_max


def patch(*pixels_with_counts):
    rows = []
    for pixel, count in pixels_with_counts:
        rows.extend([pixel] * count)
    return np.array(rows, dtype=np.uint8)


# --- circular_hue_median -----------------------------------------------------


@pytest.mark.parametrize(
    "hues, expected",
    [
        ([60] * 10, 60),
        ([2] * 5 + [177] * 5, 0),  # red straddling the boundary, not ~89 (cyan)
        ([178] * 3 + [1] * 7, 1),
        ([118, 120, 122], 120),
    ],
)
def test_circular_hue_median(hues, expected):
    result = circular_hue_median(hues)
    distance = min(abs(result - expected), 180 - abs(result - expected))
    assert distance <= 1


# --- calculate_smart_hsv_bounds ----------------------------------------------


def test_empty_patch_returns_default():
    assert calculate_smart_hsv_bounds(np.empty((0, 3)))[-1] == "Default"


def test_red_patch_split_across_hue_boundary_stays_red():
    # Regression: a plain median of [2, 177] gave H:89 (cyan) bounds 77-101.
    red = patch(([2, 200, 200], 1250), ([177, 200, 200], 1250))
    h_min, h_max, *_ , label = calculate_smart_hsv_bounds(red)
    assert label.startswith("Color")
    for h in (2, 177, 0, 179):
        assert hue_in_range(h, h_min, h_max)
    assert not hue_in_range(90, h_min, h_max)


def test_red_near_zero_wraps_to_include_high_hues():
    # Regression: median 3 used to clamp to 0-15 and miss red pixels at 171-179.
    h_min, h_max, *_ = calculate_smart_hsv_bounds(patch(([3, 200, 200], 81)))
    assert h_min > h_max  # wrapped range, handled by main.py's dual mask
    assert hue_in_range(175, h_min, h_max)
    assert hue_in_range(15, h_min, h_max)


def test_mid_hue_color_bounds():
    h_min, h_max, s_min, s_max, v_min, v_max, label = calculate_smart_hsv_bounds(
        patch(([60, 200, 150], 81))
    )
    assert (h_min, h_max) == (48, 72)
    assert (s_min, s_max, v_min, v_max) == (150, 255, 100, 210)
    assert label == "Color (H:60 S:200 V:150)"


def test_real_red_pixels_from_bgr():
    bgr = np.full((9, 9, 3), (20, 20, 220), np.uint8)
    bgr[::2, :] = (40, 10, 220)  # slightly magenta-red -> hue near 176
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h_min, h_max, *_ = calculate_smart_hsv_bounds(hsv)
    for h in set(int(v) for v in hsv[:, 0]):
        assert hue_in_range(h, h_min, h_max)


@pytest.mark.parametrize(
    "pixel, prefix",
    [
        ([0, 10, 230], "White"),
        ([0, 50, 30], "Black"),
        ([0, 10, 120], "Gray"),
    ],
)
def test_neutral_objects_use_full_hue_range(pixel, prefix):
    h_min, h_max, *_ , label = calculate_smart_hsv_bounds(patch((pixel, 25)))
    assert label.startswith(prefix)
    assert (h_min, h_max) == (0, 179)


# --- merge_nearby_boxes ------------------------------------------------------


def test_merge_empty():
    assert merge_nearby_boxes([]) == []


def test_merge_close_boxes():
    assert merge_nearby_boxes([(0, 0, 10, 10), (20, 0, 10, 10)], max_gap=15) == [(0, 0, 30, 10)]


def test_far_boxes_stay_separate():
    boxes = [(0, 0, 10, 10), (200, 200, 10, 10)]
    assert sorted(merge_nearby_boxes(boxes, max_gap=15)) == boxes


def test_merge_is_transitive():
    boxes = [(0, 0, 10, 10), (100, 0, 10, 10), (50, 0, 10, 10)]
    assert merge_nearby_boxes(boxes, max_gap=45) == [(0, 0, 110, 10)]


# --- masks and legacy helpers ------------------------------------------------


def test_clean_mask_removes_specks_and_keeps_blobs():
    mask = np.zeros((100, 100), np.uint8)
    mask[10, 10] = 255
    mask[40:70, 40:70] = 255
    cleaned = clean_mask(mask)
    assert cleaned[10, 10] == 0
    assert cleaned[55, 55] == 255


def test_get_limits_yellow_no_wrap():
    range1, range2 = get_limits([0, 255, 255])
    assert range2 is None
    assert range1[0][0] <= 30 <= range1[1][0]


def test_get_limits_red_wraps():
    range1, range2 = get_limits([0, 0, 255])
    assert range2 is not None
    assert range1[1][0] == 179 and range2[0][0] == 0


def test_get_bounds_from_sample_wraps_high_hue():
    range1, range2 = get_bounds_from_sample((175, 200, 200))
    assert range2 is not None
    assert range1[0][0] == 160 and range2[1][0] == 10

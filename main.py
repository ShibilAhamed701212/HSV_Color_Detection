import sys

import cv2
import numpy as np

from util import calculate_smart_hsv_bounds, clean_mask, merge_nearby_boxes

# --- Global Variables & App State ---
auto_mode = True            # Continuous sampling inside Target Box
color_locked = False        # Freeze/Lock current color
skin_filter_enabled = False # Filter out human skin tones
largest_only = False        # Track only largest matching object
merge_boxes_enabled = True  # Merge nearby split detections (e.g. face cut by glasses) into 1 box
target_center = None        # (x, y) center of Target Box (defaults to frame center)
active_color_name = "Auto Sampling (Place Object inside Target Box)"
current_hsv_frame = None    # Latest HSV frame, read by the mouse callback


def nothing(x):
    pass


def pick_color_event(event, x, y, flags, param):
    """
    Mouse callback: Clicking anywhere on the video frame moves the Target Box to (x, y),
    samples a 9x9 patch around (x, y), calculates smart HSV bounds, and locks onto the object.
    """
    global target_center, active_color_name, auto_mode, color_locked

    if event == cv2.EVENT_LBUTTONDOWN and current_hsv_frame is not None:
        h_img, w_img, _ = current_hsv_frame.shape
        if 0 <= x < w_img and 0 <= y < h_img:
            target_center = (x, y)

            # Sample 9x9 patch around clicked point
            half_patch = 4
            y1, y2 = max(0, y - half_patch), min(h_img, y + half_patch + 1)
            x1, x2 = max(0, x - half_patch), min(w_img, x + half_patch + 1)
            hsv_patch = current_hsv_frame[y1:y2, x1:x2].reshape(-1, 3)

            # Intelligently calculate bounds (handles White, Black, Gray, Color)
            h_min, h_max, s_min, s_max, v_min, v_max, label = calculate_smart_hsv_bounds(hsv_patch)

            cv2.setTrackbarPos("H Min", "Control Panel", h_min)
            cv2.setTrackbarPos("H Max", "Control Panel", h_max)
            cv2.setTrackbarPos("S Min", "Control Panel", s_min)
            cv2.setTrackbarPos("S Max", "Control Panel", s_max)
            cv2.setTrackbarPos("V Min", "Control Panel", v_min)
            cv2.setTrackbarPos("V Max", "Control Panel", v_max)

            auto_mode = False
            color_locked = True
            active_color_name = f"Picked: {label}"
            print(f"--> Target box moved to ({x}, {y}) | {label}")


# --- Initialize Control Panel & Trackbars ---
cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Control Panel", 420, 340)

cv2.createTrackbar("H Min", "Control Panel", 15, 179, nothing)  # type: ignore
cv2.createTrackbar("H Max", "Control Panel", 45, 179, nothing)  # type: ignore
cv2.createTrackbar("S Min", "Control Panel", 40, 255, nothing)  # type: ignore
cv2.createTrackbar("S Max", "Control Panel", 255, 255, nothing)  # type: ignore
cv2.createTrackbar("V Min", "Control Panel", 40, 255, nothing)  # type: ignore
cv2.createTrackbar("V Max", "Control Panel", 255, 255, nothing)  # type: ignore
cv2.createTrackbar("Min Area", "Control Panel", 300, 5000, nothing)  # type: ignore


def apply_preset(preset_name):
    global active_color_name, auto_mode, color_locked
    auto_mode = False
    color_locked = True
    active_color_name = preset_name
    presets = {
        "Red": (170, 10, 70, 255, 70, 255),
        "Green": (35, 85, 40, 255, 40, 255),
        "Blue": (95, 130, 50, 255, 50, 255),
        "Yellow": (20, 35, 40, 255, 40, 255),
        "Orange": (10, 25, 60, 255, 60, 255),
        "Purple": (125, 155, 50, 255, 50, 255),
        "White": (0, 179, 0, 45, 160, 255),
        "Black": (0, 179, 0, 255, 0, 60),
    }
    if preset_name in presets:
        h_min, h_max, s_min, s_max, v_min, v_max = presets[preset_name]
        cv2.setTrackbarPos("H Min", "Control Panel", h_min)
        cv2.setTrackbarPos("H Max", "Control Panel", h_max)
        cv2.setTrackbarPos("S Min", "Control Panel", s_min)
        cv2.setTrackbarPos("S Max", "Control Panel", s_max)
        cv2.setTrackbarPos("V Min", "Control Panel", v_min)
        cv2.setTrackbarPos("V Max", "Control Panel", v_max)
        print(f"--> Applied preset: {preset_name}")


print("Starting Smart HSV Color & Object Detector...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Warning: Camera index 0 could not be opened. Trying index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera. Please check your camera connection.")
    sys.exit(1)

cv2.namedWindow("HSV Color Detection")
cv2.setMouseCallback("HSV Color Detection", pick_color_event)  # type: ignore

print("\n=================== ADVANCED CONTROLS ===================")
print("  • Click on Target Object : Move target box & auto-pick color")
print("  • Spacebar              : Lock / Unlock current color")
print("  • 'a'                   : Toggle Auto-Sampling Mode On/Off")
print("  • 's'                   : Toggle Skin Filter (suppress skin tones)")
print("  • 'l'                   : Toggle Largest Object Only Mode")
print("  • 'm'                   : Toggle Box Merging (merge split fragments into 1 entity)")
print("  • '1'-'6'               : Color Presets (Red, Green, Blue, Yellow, Orange, Purple)")
print("  • '7' & '8'             : Presets for White & Black objects")
print("  • 'q'                   : Quit application")
print("=========================================================\n")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to grab frame.")
        break

    frame_h, frame_w, _ = frame.shape
    if target_center is None:
        target_center = (frame_w // 2, frame_h // 2)

    tc_x, tc_y = target_center
    box_size = 50
    half_box = box_size // 2

    # Ensure target box stays within image bounds
    tc_x = max(half_box, min(frame_w - half_box, tc_x))
    tc_y = max(half_box, min(frame_h - half_box, tc_y))

    # 1. Smooth frame to reduce noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # 2. Convert to HSV colorspace
    hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    current_hsv_frame = hsv_frame.copy()

    # 3. Smart Continuous Sampling inside Target Box
    if auto_mode and not color_locked:
        roi = hsv_frame[tc_y - half_box : tc_y + half_box, tc_x - half_box : tc_x + half_box]
        patch_pixels = roi.reshape(-1, 3)

        h_min, h_max, s_min, s_max, v_min, v_max, label = calculate_smart_hsv_bounds(patch_pixels)

        cv2.setTrackbarPos("H Min", "Control Panel", h_min)
        cv2.setTrackbarPos("H Max", "Control Panel", h_max)
        cv2.setTrackbarPos("S Min", "Control Panel", s_min)
        cv2.setTrackbarPos("S Max", "Control Panel", s_max)
        cv2.setTrackbarPos("V Min", "Control Panel", v_min)
        cv2.setTrackbarPos("V Max", "Control Panel", v_max)

        active_color_name = f"Auto: {label}"

    # 4. Read trackbar positions
    h_min = cv2.getTrackbarPos("H Min", "Control Panel")
    h_max = cv2.getTrackbarPos("H Max", "Control Panel")
    s_min = cv2.getTrackbarPos("S Min", "Control Panel")
    s_max = cv2.getTrackbarPos("S Max", "Control Panel")
    v_min = cv2.getTrackbarPos("V Min", "Control Panel")
    v_max = cv2.getTrackbarPos("V Max", "Control Panel")
    min_area = max(50, cv2.getTrackbarPos("Min Area", "Control Panel"))

    # 5. Generate binary mask
    if h_min <= h_max:
        lower_bound = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_bound = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    else:
        lower1 = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper1 = np.array([179, s_max, v_max], dtype=np.uint8)
        lower2 = np.array([0, s_min, v_min], dtype=np.uint8)
        upper2 = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask1 = cv2.inRange(hsv_frame, lower1, upper1)
        mask2 = cv2.inRange(hsv_frame, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    # Optional Skin Suppress Filter
    if skin_filter_enabled:
        skin_lower = np.array([0, 25, 60], dtype=np.uint8)
        skin_upper = np.array([25, 140, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv_frame, skin_lower, skin_upper)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))

    # Clean mask with morphological ops
    mask = clean_mask(mask, kernel_size=5)

    # 6. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if largest_only and len(valid_contours) > 0:
        valid_contours = [max(valid_contours, key=cv2.contourArea)]

    raw_boxes = [cv2.boundingRect(c) for c in valid_contours]

    if merge_boxes_enabled and not largest_only and len(raw_boxes) > 0:
        final_boxes = merge_nearby_boxes(raw_boxes, max_gap=50)
    else:
        final_boxes = raw_boxes

    detected_count = len(final_boxes)

    for (x, y, w, h) in final_boxes:
        area = w * h

        # Draw bounding rectangle around detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label_str = f"{active_color_name} ({int(area)}px)"
        cv2.putText(
            frame,
            label_str,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        cx_c = x + w // 2
        cy_c = y + h // 2
        cv2.circle(frame, (cx_c, cy_c), 4, (0, 0, 255), -1)

    # --- Draw Target Box ---
    box_color = (0, 255, 255) if auto_mode and not color_locked else (0, 165, 255)
    cv2.rectangle(
        frame,
        (tc_x - half_box, tc_y - half_box),
        (tc_x + half_box, tc_y + half_box),
        box_color,
        2,
    )
    cv2.circle(frame, (tc_x, tc_y), 3, box_color, -1)

    # --- Draw HUD ---
    overlay_bg = frame.copy()
    cv2.rectangle(overlay_bg, (10, 10), (520, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bg, 0.65, frame, 0.35, 0, frame)

    mode_status = "LOCKED" if color_locked else ("AUTO" if auto_mode else "MANUAL")
    skin_str = "ON" if skin_filter_enabled else "OFF"
    largest_str = "ON" if largest_only else "OFF"
    merge_str = "ON" if merge_boxes_enabled else "OFF"

    status_color = (0, 255, 0) if color_locked else ((0, 255, 255) if auto_mode else (200, 200, 200))

    cv2.putText(frame, f"MODE: [{mode_status}] | Found: {detected_count} | SkinFilter: [{skin_str}] | Largest: [{largest_str}] | Merge: [{merge_str}]", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 2)
    cv2.putText(frame, f"Target: {active_color_name}", (20, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 200), 1)
    cv2.putText(frame, f"H:[{h_min}-{h_max}] S:[{s_min}-{s_max}] V:[{v_min}-{v_max}] MinArea:{min_area}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1)
    cv2.putText(frame, "Click object | SPACE: Lock | 's': Skin | 'l': Largest | 'm': Merge | 'q': Quit", (20, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

    # Show live video frame and binary mask
    cv2.imshow("HSV Color Detection", frame)
    cv2.imshow("Binary Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Quitting application...")
        break
    elif key == ord(" "):
        color_locked = not color_locked
        unlocked_str = "UNLOCKED (Auto Sampling)" if auto_mode else "UNLOCKED (Manual)"
        print(f"--> Color Lock: {'LOCKED' if color_locked else unlocked_str}")
    elif key == ord("a"):
        auto_mode = not auto_mode
        color_locked = False
        print(f"--> Auto Sampling Mode: {'ENABLED' if auto_mode else 'DISABLED'}")
    elif key == ord("s"):
        skin_filter_enabled = not skin_filter_enabled
        print(f"--> Skin Suppress Filter: {'ENABLED' if skin_filter_enabled else 'DISABLED'}")
    elif key == ord("l"):
        largest_only = not largest_only
        print(f"--> Track Largest Only: {'ENABLED' if largest_only else 'DISABLED'}")
    elif key == ord("m"):
        merge_boxes_enabled = not merge_boxes_enabled
        print(f"--> Merge Nearby Boxes: {'ENABLED' if merge_boxes_enabled else 'DISABLED'}")
    elif key == ord("1"):
        apply_preset("Red")
    elif key == ord("2"):
        apply_preset("Green")
    elif key == ord("3"):
        apply_preset("Blue")
    elif key == ord("4"):
        apply_preset("Yellow")
    elif key == ord("5"):
        apply_preset("Orange")
    elif key == ord("6"):
        apply_preset("Purple")
    elif key == ord("7"):
        apply_preset("White")
    elif key == ord("8"):
        apply_preset("Black")

cap.release()
cv2.destroyAllWindows()
print("Application stopped cleanly.")
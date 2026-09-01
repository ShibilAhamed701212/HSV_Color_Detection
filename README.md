# 🎯 Real-Time Smart HSV Color & Object Detector

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-green.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.23%2B-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

An intelligent, real-time Computer Vision application built with **OpenCV** and **Python** that detects, tracks, and isolates objects based on color in HSV color space. Featuring automated color sampling, interactive mouse-click calibration, smart boundary calculation for vibrant and neutral/monochrome objects, skin-tone suppression, and fragmented bounding-box merging.

---

## ✨ Features

- 🎯 **Interactive Point-and-Click Sampling:** Click on any object in the live video stream to instantly move the target box, calculate optimal HSV boundaries, and lock tracking.
- ⚡ **Auto Dynamic Sampling:** Continuously samples pixels within the target reticle in real-time to adjust Hue, Saturation, and Value thresholds dynamically.
- 🧠 **Smart Color Analysis:** Intelligent thresholding logic capable of distinguishing saturated colors (Red, Blue, Green, Yellow, etc.) from neutral tones (White, Black, Gray).
- 🔄 **Hue Wraparound Support:** Seamlessly handles colors spanning the 0°/180° Hue boundary (e.g., Red) using dual-range masking.
- 🛡️ **Skin Tone Filter:** Suppresses human skin color ranges to prevent false-positive object tracking around hands and faces.
- 📦 **Bounding Box Merging:** Automatically merges fragmented contour bounding boxes caused by reflections, occlusion, or shadows into unified object detections.
- 🎛️ **Live Control Panel & Sliders:** Dedicated OpenCV trackbar GUI for real-time manual fine-tuning of HSV boundaries and minimum contour area thresholds.
- 🎨 **Quick-Color Preset Keys:** 1-key shortcuts for Red, Green, Blue, Yellow, Orange, Purple, White, and Black.
- 📊 **Real-Time HUD Overlay:** Semi-transparent heads-up display showing active mode, object counts, color labels, and HSV threshold coordinates.

---

## 🎮 Controls & Shortcuts

| Action / Key | Function |
| :--- | :--- |
| **Left Click** | Move sampling box to clicked pixel, calculate smart bounds, and lock color |
| **`Spacebar`** | Toggle Lock / Unlock current HSV bounds |
| **`a`** | Toggle Continuous Auto-Sampling Mode |
| **`s`** | Toggle Skin Suppression Filter (suppress skin tones) |
| **`l`** | Toggle Largest Object Only tracking mode |
| **`m`** | Toggle Bounding Box Merging (unify split contours) |
| **`1` - `6`** | Load Presets: `1`=Red, `2`=Green, `3`=Blue, `4`=Yellow, `5`=Orange, `6`=Purple |
| **`7` & `8`** | Load Presets: `7`=White, `8`=Black |
| **`q`** | Quit the application |

---

## 🛠️ Project Structure

```text
HSV_Color_Detection/
├── main.py              # Main application loop, camera capture, HUD, and GUI events
├── util.py              # HSV calculations, smart boundary analysis, mask cleaning, box merging
├── requirements.txt     # Python dependencies (opencv-python, numpy, pillow)
├── .gitignore           # Git ignore rules for virtual environments and cache
└── README.md            # Project documentation and guide
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ShibilAhamed701212/HSV_Color_Detection.git
cd HSV_Color_Detection
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

---

## 🔬 How It Works

1. **Color Space Transformation:** Converts incoming BGR webcam frames into the **HSV (Hue, Saturation, Value)** color space, which separates chromatic information from lighting intensity.
2. **Morphological Cleaning:** Applies morphological **Opening** (erosion followed by dilation) to eliminate background noise specks and **Closing** (dilation followed by erosion) to fill interior holes.
3. **Contour Extraction & Filtering:** Extracts external contours and filters out noise smaller than the user-defined `Min Area` trackbar value.
4. **Nearby Box Clustering:** Groups bounding boxes within a defined pixel distance to maintain stable bounding rectangles around complex objects.

---

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute for personal and educational projects.

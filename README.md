
# Vehicle Counter (YOLO + ByteTrack)

A compact project that counts vehicles crossing user-defined lines in a video using a YOLO model for detection and ByteTrack for tracking. The interface lets you draw one or more lines on a displayed frame, then the script processes the video and produces an annotated output with per-line and per-class counts.

# Features

- Interactive drawing of multiple counting lines on a frame.
- Detection + tracking (YOLOv12 / ultralytics + ByteTrack tracker).
- Per-line counts (total) and per-line breakdown by vehicle class (car, motorcycle, bus, truck).
- Colored lines and matching-colored post-crossing bounding boxes for clarity.
- Vehicle ID display for better tracking visualization.
- Scales selection frame for high-resolution videos.
- Export annotated video (MP4) with counters overlay.

# Repository — clone & run

## 1. clone the repository
```bash
git clone https://github.com/yanal154/vehicles_counter
cd vehicles_counter
```

## 2. create and activate a venv (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

## 3. install requirements

```bash
pip install -r requirements.txt
```

## 4. Download YOLOv12 Model

Before running the project, you need to download the YOLOv12 weights file (`yolo12l.pt`) from the following link:

[Download YOLOv12l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt)

After downloading, place the file in the project folder or update the file path in the code:

```python
from ultralytics import YOLO

MODEL_WEIGHTS = "yolo12l.pt"  # Update the path if the file is in a different location
model = YOLO(MODEL_WEIGHTS)
```

## 5. put your video file in the repo and set VIDEO_PATH in the script

## 6. run

```bash
python main.py
```

# Configuration

* **VIDEO_PATH** — path to input video (set in `main.py`).
* **OUTPUT_VIDEO** — output annotated filename.
* **MODEL_WEIGHTS** — YOLO weights file (e.g. `yolo12l.pt`) or local path. Place the weight file in the repo or provide a full path.
* **bytetrack.yaml** — ByteTrack tracker config should be present (or use the default provided by ultralytics). Ensure `lap` is installed for tracker matching.

# How it works (brief)

1. Script loads the first frame of the video and displays it scaled (for interactive convenience).
2. User clicks twice for each line (repeat for multiple lines). Press:

   * `s` to start processing
   * `r` to reset selections
   * `Esc` to cancel
3. Script converts clicked coordinates back to original video resolution, then runs the model with `model.track(..., stream=True)`.
4. For each tracked object of vehicle classes, it checks if its centroid crosses any line (side change + projection on segment).
5. When a crossing is detected (and debounced by a minimum frame gap), it increments the per-line and per-class counters and temporarily colors the bbox with the line color.
6. Vehicle IDs are displayed for better tracking visualization and debugging.
7. An annotated output video is written and counts are printed to console.

from ultralytics import YOLO
import cv2
import os
import math
import time
import numpy as np

VIDEO_PATH = "14350554_3840_2160_60fps.mp4"     
OUTPUT_VIDEO = "VEHICLE_counter.mp4"
MODEL_WEIGHTS = "yolo12l.pt"   

model = YOLO(MODEL_WEIGHTS)


line_points_list = []

window_name = "select_lines"

VEHICLE_CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def mouse_callback(event, x, y, flags, param):
    """Click twice to draw a new line (coordinates are on the displayed scaled image)."""
    global line_points_list
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points_list) == 0 or len(line_points_list[-1]) == 2:
            line_points_list.append([(x, y)])
            print(f"Start new line at {(x, y)} (display coords)")
        elif len(line_points_list[-1]) == 1:
            line_points_list[-1].append((x, y))
            print(f"End line at {(x, y)} (display coords)")


def hsv_to_bgr_tuple(h, s=255, v=255):
    """Return BGR tuple from HSV (h in 0..179)."""
    hsv = np.uint8([[[h, s, v]]]) 
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return tuple(int(c) for c in bgr)


def get_line_colors(n):
    """Generate n visually-distinct BGR colors."""
    colors = []
    for i in range(n):
        # spread hues across 0..179, avoid extremes that are too dark/light
        h = int((i * 160) / max(1, n - 1))  # 0..160
        colors.append(hsv_to_bgr_tuple(h, 200, 255))
    return colors


def select_lines_interactive(video_path, display_max_w=800):
    """
    Select multiple lines on a single frame.
    Returns a list of lines with coordinates in the original video size.
    """
    global line_points_list
    line_points_list = []

    cap = cv2.VideoCapture(video_path)
    ret, frame_orig = cap.read()
    cap.release()
    if not ret:
        print("❌ Cannot read video")
        return None

    orig_h, orig_w = frame_orig.shape[:2]
    frame = frame_orig.copy()
    scale = 1.0
    if orig_w > display_max_w:
        scale = display_max_w / orig_w
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        frame = cv2.resize(frame_orig, (new_w, new_h))

    print("Click twice to draw a line, repeat for more lines")
    print("Controls: 's' start, 'r' reset, 'Esc' quit")

    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = frame.copy()
        for line in line_points_list:
            for (px, py) in line:
                cv2.circle(temp, (px, py), 6, (0, 255, 0), -1)
            if len(line) == 2:
                cv2.line(temp, line[0], line[1], (0, 255, 0), 2)

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc
            print("Esc pressed. Exiting.")
            cv2.destroyAllWindows()
            return None
        if key == ord('r'):
            line_points_list = []
            print("Reset all lines.")
        if key == ord('s'):
            if not line_points_list or any(len(l) < 2 for l in line_points_list):
                print("❌ Draw full lines before starting.")
                continue
            # convert coordinates from display to original video coordinates
            scaled_lines = []
            for line in line_points_list:
                p1_display, p2_display = line[0], line[1]
                x1_orig = int(p1_display[0] / scale)
                y1_orig = int(p1_display[1] / scale)
                x2_orig = int(p2_display[0] / scale)
                y2_orig = int(p2_display[1] / scale)
                # clamp to image bounds
                x1_orig = max(0, min(orig_w - 1, x1_orig))
                x2_orig = max(0, min(orig_w - 1, x2_orig))
                y1_orig = max(0, min(orig_h - 1, y1_orig))
                y2_orig = max(0, min(orig_h - 1, y2_orig))
                scaled_lines.append(((x1_orig, y1_orig), (x2_orig, y2_orig)))
            cv2.destroyAllWindows()
            print(f"Selected lines (original coords): {scaled_lines}")
            return scaled_lines


def point_side_of_line(px, py, x1, y1, x2, y2):
    """
    Return the side of a point relative to the directed line from (x1,y1) to (x2,y2).
    We use cross value z = dx*(py - y1) - dy*(px - x1).
    Return 1 if Z>0, -1 if Z<0, 0 if near zero.
    """
    dx = x2 - x1
    dy = y2 - y1
    z = dx * (py - y1) - dy * (px - x1)
    if z > 0:
        return 1
    if z < 0:
        return -1
    return 0


def expand_bbox(x1, y1, x2, y2, img_w, img_h, pad_ratio=0.05):
    """Return expanded bbox clamped to image, pad_ratio fraction of width/height."""
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(img_w - 1, x2 + pad_x)
    ny2 = min(img_h - 1, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def count_vehicle_with_lines(input_path, output_path, lines, max_frames=None):
    if not os.path.exists(input_path):
        print("❌ File not found:", input_path)
        return

    n_lines = len(lines)
    line_colors = get_line_colors(n_lines)

    print(f"▶️ Start counting vehicles in {input_path} (lines={n_lines})")

    results_gen = model.track(
        source=input_path,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
        classes=list(VEHICLE_CLASS_MAP.keys())   # only vehicle classes
    )

    out = None
    frame_idx = 0

    # per-line counts by vehicle type + total
    counts_by_line = []
    for _ in range(n_lines):
        d = {v: 0 for v in VEHICLE_CLASS_MAP.values()}
        d['total'] = 0
        counts_by_line.append(d)

    # state for each track per line
    track_sides = {i: {} for i in range(n_lines)}
    last_count_frame = {}  # (track_id, line_idx) -> last frame counted
    MIN_FRAMES_BETWEEN_COUNTS = 5

    # remember last crossed line for coloring bbox and decay after some frames
    last_line_for_track = {}       # track_id -> (line_idx, frame_idx)
    LINE_COLOR_DECAY_FRAMES = 30   # keep coloring bbox for this many frames after crossing

    for result in results_gen:
        frame = result.orig_img
        if frame is None:
            continue

        if out is None:
            h, w = frame.shape[:2]
            fps = 30
            cap_check = cv2.VideoCapture(input_path)
            f = cap_check.get(cv2.CAP_PROP_FPS)
            cap_check.release()
            if f and f > 0:
                fps = f
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            print(f"Output video: {output_path} size={(w,h)} fps={fps}")

        # draw lines with distinct colors
        for idx, (p1, p2) in enumerate(lines):
            color = line_colors[idx]
            cv2.line(frame, p1, p2, color, 4)
            # draw small colored midpoint circle
            mx = (p1[0] + p2[0]) // 2
            my = (p1[1] + p2[1]) // 2
            cv2.circle(frame, (mx, my), 8, color, -1)
            # draw small label rectangle with color top-left area later

        if result.boxes is None or len(result.boxes) == 0:
            # draw legend even when no detections
            for i, cnts in enumerate(counts_by_line):
                color = line_colors[i]
                base_x = 20
                base_y = 20 + i * 70
                rect_w = 420
                rect_h = 60
                # colored strip at left of legend
                cv2.rectangle(frame, (base_x, base_y), (base_x + 18, base_y + rect_h), color, -1)
                # background
                cv2.rectangle(frame, (base_x + 20, base_y), (base_x + 20 + rect_w, base_y + rect_h), (0, 0, 0), -1)
                cv2.putText(frame, f"Line {i+1} Total: {cnts['total']}", (base_x + 28, base_y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                breakdown = f"car:{cnts['car']}  bus:{cnts['bus']}  truck:{cnts['truck']}  moto:{cnts['motorcycle']}"
                cv2.putText(frame, breakdown, (base_x + 28, base_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            out.write(frame)
            frame_idx += 1
            continue

        # get track ids if available
        try:
            ids = result.boxes.id.int().cpu().tolist()
        except Exception:
            ids = [None] * len(result.boxes)

        bboxes = result.boxes.xyxy.cpu().tolist()
        cls_ids = result.boxes.cls.int().cpu().tolist()

        for det_id, bbox, cls_id in zip(ids, bboxes, cls_ids):
            if cls_id not in VEHICLE_CLASS_MAP:
                continue

            x1o, y1o, x2o, y2o = map(int, bbox)
            # expand bbox slightly for bigger visual boxes
            nx1, ny1, nx2, ny2 = expand_bbox(x1o, y1o, x2o, y2o, frame.shape[1], frame.shape[0], pad_ratio=0.06)
            cx = (nx1 + nx2) // 2
            cy = (ny1 + ny2) // 2

            # default bbox color
            bbox_color = (0, 255, 0)
            bbox_thickness = 3

            # if this track recently crossed a line, color bbox with that line color
            track_id = None
            if det_id is not None:
                track_id = int(det_id)
                last = last_line_for_track.get(track_id, None)
                if last is not None:
                    line_idx_crossed, frame_of_cross = last
                    if frame_idx - frame_of_cross <= LINE_COLOR_DECAY_FRAMES:
                        bbox_color = line_colors[line_idx_crossed]
                        bbox_thickness = 4

            # draw bbox with possibly line color
            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), bbox_color, bbox_thickness)
            cv2.circle(frame, (cx, cy), 5, bbox_color, -1)

            vehicle_type = VEHICLE_CLASS_MAP[cls_id]

            # if no track id, skip counting (can't avoid duplicates reliably)
            if track_id is None:
                # put small label in white
                cv2.putText(frame, vehicle_type, (nx1, max(15, ny1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                continue

            # check each line for crossing (based on side change & projection on segment)
            for i, ((x1l, y1l), (x2l, y2l)) in enumerate(lines):
                dxl = x2l - x1l
                dyl = y2l - y1l
                seg_len2 = dxl * dxl + dyl * dyl
                if seg_len2 == 0:
                    continue
                # projection parameter t (allow slight margin)
                t = ((cx - x1l) * dxl + (cy - y1l) * dyl) / seg_len2
                if t < -0.1 or t > 1.1:
                    continue

                side = point_side_of_line(cx, cy, x1l, y1l, x2l, y2l)
                prev = track_sides[i].get(track_id, None)
                if prev is None:
                    track_sides[i][track_id] = side
                else:
                    if side != prev and side != 0:
                        key = (track_id, i)
                        last = last_count_frame.get(key, -9999)
                        if frame_idx - last >= MIN_FRAMES_BETWEEN_COUNTS:
                            # count by vehicle type and total
                            counts_by_line[i][vehicle_type] += 1
                            counts_by_line[i]['total'] += 1
                            last_count_frame[key] = frame_idx
                            track_sides[i][track_id] = side
                            last_line_for_track[track_id] = (i, frame_idx)
                            print(f"[f{frame_idx}] track {track_id} ({vehicle_type}) crossed line {i+1} -> total={counts_by_line[i]['total']}")
                        else:
                            track_sides[i][track_id] = side
                    else:
                        track_sides[i][track_id] = side

            # label vehicle type above bbox
            cv2.putText(frame, vehicle_type, (nx1, max(15, ny1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # draw counters on video (per class) with colored strip per line
        base_x = 20
        base_y = 20
        line_gap = 90
        for i, cnts in enumerate(counts_by_line):
            color = line_colors[i]
            # colored strip on left
            rx = base_x
            ry = base_y + i * line_gap
            rect_w = 460
            rect_h = 70
            cv2.rectangle(frame, (rx, ry), (rx + 18, ry + rect_h), color, -1)
            # background box
            cv2.rectangle(frame, (rx + 20, ry), (rx + 20 + rect_w, ry + rect_h), (0, 0, 0), -1)
            # title with same color
            cv2.putText(frame, f"Line {i+1} Total: {cnts['total']}", (rx + 28, ry + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            breakdown = f"car:{cnts['car']}  bus:{cnts['bus']}  truck:{cnts['truck']}  moto:{cnts['motorcycle']}"
            cv2.putText(frame, breakdown, (rx + 28, ry + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    if out:
        out.release()
    print("✅ Finished. Counts by line:")
    for i, c in enumerate(counts_by_line):
        print(f" Line {i+1}: {c}")
    return counts_by_line


if __name__ == "__main__":
    lines = select_lines_interactive(VIDEO_PATH, display_max_w=800)
    if lines is None:
        print("No lines selected. Exiting.")
        exit()

    counts = count_vehicle_with_lines(
        VIDEO_PATH,
        output_path=OUTPUT_VIDEO,
        lines=lines,
        max_frames=None
    )

    print("Final counts:", counts)
    try:
        if os.name == 'nt':
            os.startfile(OUTPUT_VIDEO)
        else:
            os.system(f'xdg-open "{OUTPUT_VIDEO}"')
    except Exception:
        pass

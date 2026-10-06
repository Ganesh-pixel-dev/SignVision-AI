"""
player.py
----------
RUN THIS ON YOUR OWN MACHINE.

Loads a recorded phrase from skeleton/library/<PHRASE>.json and plays
it back as an animated stick-figure skeleton (face outline, arms,
both hands) drawn with OpenCV - a quick way to preview recordings
without needing the full web app.

Usage:
    python player.py "HELLO"
"""

import json
import os
import sys
import time

import numpy as np
import cv2

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")

# Minimal hand connections (MediaPipe Hands topology) for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# A few key pose connections: shoulders -> elbows -> wrists (arms only, matches the mockup)
POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # left shoulder->elbow->wrist
    (12, 14), (14, 16),  # right shoulder->elbow->wrist
    (11, 12),            # shoulder to shoulder
]

CANVAS_SIZE = (720, 720)


def to_pixel(point, w, h):
    return int(point[0] * w), int(point[1] * h)


def draw_points(canvas, points, connections, color, w, h, radius=3):
    if points is None:
        return
    pts = [to_pixel(p, w, h) for p in points]
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(canvas, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(canvas, p, radius, color, -1)


def draw_face_outline(canvas, face_points, color, w, h):
    """Simplified: draw the face as a sparse outline rather than the full 468-point mesh,
    to visually match a clean stick-skeleton look."""
    if face_points is None:
        return
    # Just draw the outer face oval (MediaPipe face mesh has a known oval contour subset)
    oval_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    pts = [to_pixel(face_points[i], w, h) for i in oval_idx if i < len(face_points)]
    for i in range(len(pts)):
        cv2.line(canvas, pts[i], pts[(i + 1) % len(pts)], color, 1)


def main():
    if len(sys.argv) < 2:
        print('Usage: python player.py "PHRASE_NAME"')
        return

    phrase = sys.argv[1].strip().upper()
    path = os.path.join(LIBRARY_DIR, f"{phrase}.json")
    if not os.path.exists(path):
        print(f"No recording found for '{phrase}' at {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    frames = data["frames"]
    fps = data.get("fps", 30)
    delay = 1.0 / fps if fps > 0 else 1.0 / 30
    w, h = CANVAS_SIZE

    print(f"Playing '{phrase}' ({len(frames)} frames @ {fps:.1f} fps). Press 'q' to quit, any other key to loop again.")

    while True:
        for fr in frames:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            draw_face_outline(canvas, fr.get("face"), (0, 0, 220), w, h)
            draw_points(canvas, fr.get("pose"), POSE_CONNECTIONS, (0, 0, 220), w, h)
            draw_points(canvas, fr.get("left_hand"), HAND_CONNECTIONS, (0, 165, 255), w, h)
            draw_points(canvas, fr.get("right_hand"), HAND_CONNECTIONS, (255, 0, 0), w, h)

            cv2.putText(canvas, phrase, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("Skeleton Playback", canvas)
            key = cv2.waitKey(int(delay * 1000)) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
        time.sleep(0.5)  # brief pause before looping


if __name__ == "__main__":
    main()

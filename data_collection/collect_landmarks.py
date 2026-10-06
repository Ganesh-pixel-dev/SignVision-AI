"""
collect_landmarks.py
---------------------
RUN THIS ON YOUR OWN MACHINE (needs a real webcam).

Lets you record hand-landmark samples for each gesture class by holding
up the sign and pressing its key. Saves everything into a single CSV
that training/train_model.py will use.

Controls:
    - Press any class key listed in CLASSES (e.g. 'a' for A) to save the
      CURRENT frame's landmark vector under that label.
    - Hold the key down / tap repeatedly to gather many samples quickly.
    - Press 'q' to quit.

Recommended: ~150-300 samples per class, varying hand angle, distance,
and lighting a little for each.

Start small: begin with CLASSES = list("ABCDEFGHIJ") (10 letters) to
get the full pipeline working before expanding to more.
"""

import csv
import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.hand_tracker import HandTracker

# ---- CONFIG ----
CLASSES = list("ABCDEFGHIJ")  # start with 10 classes, expand later
OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "dataset", "landmarks.csv",
)
# ----------------


def ensure_csv_header(path, num_features=63):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"f{i}" for i in range(num_features)] + ["label"]
            writer.writerow(header)


def main():
    ensure_csv_header(OUTPUT_CSV)
    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions/index.")
        return

    print("=" * 60)
    print("SignVision AI - Landmark Data Collector")
    print("Classes:", ", ".join(CLASSES))
    print("Press the letter key to save a sample of that gesture.")
    print("Press 'q' to quit.")
    print("=" * 60)

    counts = {c: 0 for c in CLASSES}

    # Pre-count existing samples so the on-screen counter is accurate
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row.get("label")
                if lbl in counts:
                    counts[lbl] += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        results = tracker.process(frame)
        frame = tracker.draw_landmarks(frame, results)

        # UI overlay
        y = 30
        cv2.putText(frame, "Press a class key to save sample | 'q' to quit",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 30
        counts_text = "  ".join(f"{c}:{counts[c]}" for c in CLASSES)
        cv2.putText(frame, counts_text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("SignVision AI - Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        pressed_char = chr(key).upper() if 0 <= key < 256 else ""
        if pressed_char in CLASSES:
            feature_vector = HandTracker.extract_feature_vector(results)
            if feature_vector is not None:
                with open(OUTPUT_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(feature_vector) + [pressed_char])
                counts[pressed_char] += 1
                print(f"Saved sample for '{pressed_char}' (total: {counts[pressed_char]})")
            else:
                print("No hand detected - sample not saved.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone. Dataset saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()

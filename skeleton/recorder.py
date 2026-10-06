"""
recorder.py
------------
RUN THIS ON YOUR OWN MACHINE (needs a real webcam).

Records a sign-language phrase as a TIME SEQUENCE of skeleton
landmarks (face, pose/arms, both hands) using MediaPipe Holistic.
This is the "speech -> sign" side: instead of trying to generate new
sign animations with AI (an unsolved research problem), we record a
real person performing common words/phrases once, and play the
recording back as an animated skeleton whenever that word is needed.

Usage:
    python recorder.py "HELLO"

Controls:
    - Press SPACE to start recording, SPACE again to stop and save.
    - Press 'q' to quit without saving current take.

Output:
    skeleton/library/<PHRASE>.json
    Format: {
        "phrase": "HELLO",
        "fps": 30,
        "frames": [
            {"face": [[x,y,z], ...] or null,
             "pose": [[x,y,z], ...] or null,
             "left_hand": [[x,y,z], ...] or null,
             "right_hand": [[x,y,z], ...] or null},
            ...
        ]
    }
"""

import json
import os
import sys
import time

import cv2
import mediapipe as mp

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")


def landmarks_to_list(landmark_list):
    if landmark_list is None:
        return None
    return [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark]


def main():
    if len(sys.argv) < 2:
        print('Usage: python recorder.py "PHRASE_NAME"  (e.g. "HELLO")')
        return

    phrase = sys.argv[1].strip().upper()
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    out_path = os.path.join(LIBRARY_DIR, f"{phrase}.json")

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print(f"Ready to record phrase: '{phrase}'")
    print("Press SPACE to start recording, SPACE again to stop & save. 'q' to cancel.")

    recording = False
    frames = []
    start_time = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # draw for visual feedback
        mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        status = "RECORDING..." if recording else "Press SPACE to start"
        color = (0, 0, 255) if recording else (0, 255, 255)
        cv2.putText(frame, f"Phrase: {phrase} | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if recording:
            cv2.putText(frame, f"Frames captured: {len(frames)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("SignVision AI - Skeleton Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not recording:
                recording = True
                frames = []
                start_time = time.time()
                print("Recording started...")
            else:
                recording = False
                elapsed = time.time() - start_time
                fps = len(frames) / elapsed if elapsed > 0 else 30
                data = {"phrase": phrase, "fps": round(fps, 2), "frames": frames}
                with open(out_path, "w") as f:
                    json.dump(data, f)
                print(f"Saved {len(frames)} frames ({elapsed:.1f}s) to {out_path}")
                break

        if recording:
            frames.append({
                "face": landmarks_to_list(results.face_landmarks),
                "pose": landmarks_to_list(results.pose_landmarks),
                "left_hand": landmarks_to_list(results.left_hand_landmarks),
                "right_hand": landmarks_to_list(results.right_hand_landmarks),
            })

        if key == ord("q"):
            print("Cancelled - nothing saved.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

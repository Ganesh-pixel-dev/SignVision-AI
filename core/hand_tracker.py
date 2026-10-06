"""
hand_tracker.py
----------------
Thin wrapper around MediaPipe Hands.

Why landmarks instead of raw images?
- MediaPipe gives us 21 (x, y, z) points per hand = 63 numbers.
- Training a classifier on these 63 numbers is faster, needs far less
  data, and is more robust to lighting/background than training a CNN
  on raw pixel images.
- We still get to call this "deep learning" honestly: the classifier
  itself (see core/gesture_classifier.py) is a small neural network.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def process(self, frame_bgr):
        """
        Runs hand detection on a single BGR frame (as read by OpenCV).
        Returns the raw MediaPipe results object.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        return results

    def draw_landmarks(self, frame_bgr, results):
        """Draws the hand skeleton on top of the frame (for on-screen debugging/demo)."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    @staticmethod
    def extract_feature_vector(results):
        """
        Converts MediaPipe's hand landmarks into a flat, normalized
        63-number vector (21 points x, y, z) ready for the classifier.

        Normalization strategy:
        - Translate so the wrist (landmark 0) is the origin.
        - Scale so the distance from wrist to middle-finger MCP (landmark 9)
          is 1.0.
        This makes the vector roughly invariant to hand position on
        screen and distance from the camera.

        Returns None if no hand was detected.
        """
        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
        )

        wrist = points[0].copy()
        points -= wrist  # translate

        scale_ref = np.linalg.norm(points[9])  # wrist -> middle finger MCP
        if scale_ref < 1e-6:
            scale_ref = 1e-6
        points /= scale_ref  # scale

        return points.flatten()  # shape (63,)

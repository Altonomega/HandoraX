"""Hand detection via the MediaPipe Tasks HandLandmarker.

This replaces the old ``mp.solutions.hands`` API, which MediaPipe
deprecated in favour of Tasks. The detector returns plain landmark lists
(a list of hands, each a list of 21 points with .x/.y), so the rest of
the program never sees a MediaPipe type.

The model file (``models/hand_landmarker.task``) is not committed to the
repo; it is downloaded from Google's official model store on first use
(see :func:`ensure_model`).
"""

import os
import time
import urllib.request

from config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    MODEL_URL,
)


def ensure_model(model_path=MODEL_PATH, model_url=MODEL_URL):
    """Download the HandLandmarker model if it is not already present."""
    if os.path.exists(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"Downloading hand landmarker model to {model_path} ...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except Exception as exc:  # noqa: BLE001 - report and re-raise plainly
        raise RuntimeError(
            f"Could not download the hand landmarker model from "
            f"{model_url}. Check your connection and try again."
        ) from exc
    print("Model downloaded.")
    return model_path


class HandDetector:
    def __init__(
        self,
        model_path=None,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is not installed. Run: pip install -r requirements.txt"
            ) from exc
        self._mp = mp
        model_path = ensure_model(model_path or MODEL_PATH)
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )
        self._last_timestamp_ms = 0

    def detect(self, frame, timestamp_ms=None):
        """Detect hands in a BGR frame (as returned by OpenCV).

        Returns a list of hands (each a list of 21 landmarks with .x/.y),
        or None when no hand is present.
        """
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb_frame
        )
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        # detect_for_video requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.hand_landmarks:
            return result.hand_landmarks
        return None

    def close(self):
        self.landmarker.close()

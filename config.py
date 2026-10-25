# HandoraX configuration.
#
# All tunables live here. Nothing in gesture_recognizer.py or
# system_controller.py hardcodes a threshold anymore, so behaviour can be
# tuned (and tested) without touching logic.

# --- Detection (MediaPipe Tasks HandLandmarker) ---
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "models/hand_landmarker.task"
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- Gestures ---
# A pinch counts when fingertip-to-thumb distance is below this fraction of
# the hand's own size (wrist -> middle-finger knuckle). Relative, not
# absolute, so the same gesture works whether the hand is near or far from
# the camera. The old code used a fixed 0.07 in normalized coordinates.
PINCH_RATIO = 0.35
# A finger counts as extended when its tip is this much farther from the
# wrist than its middle joint is. Orientation-independent: unlike the old
# "tip above wrist" test, this works with a tilted or sideways hand.
FINGER_EXTENDED_RATIO = 1.15
# Pinch held longer than this becomes a hold (drag) instead of a click.
CLICK_HOLD_SECONDS = 0.3
# Thumbs-up toggles pause; toggles closer than this are ignored so one
# gesture cannot flip pause on and off in the same second.
PAUSE_DEBOUNCE_SECONDS = 1.0
# Calibration averages this many seconds of palm samples instead of
# trusting the last frame.
CALIBRATION_SECONDS = 3.0

# --- Active zone (comfortable hand range -> full screen) ---
# The old code mapped the whole camera frame 1:1 onto the screen, so the
# hand had to cross the entire frame (and often left it) to cross the
# screen. Instead, calibration measures the user's comfortable hand
# range and maps THAT zone onto the full screen: a small, relaxed hand
# movement covers every pixel, and the hand never needs to leave the
# frame. Positions outside the zone clamp to the screen edge.
CALIBRATION_FILE = "calibration.json"
# Phase 2 of calibration: seconds of free hand movement inside the
# comfortable range.
RANGE_CALIBRATION_SECONDS = 5.0
# Outlier trimming for the range: the 2nd/98th percentiles of the
# sampled palm positions, so a stray frame doesn't stretch the zone.
ZONE_PERCENTILE_LOW = 0.02
ZONE_PERCENTILE_HIGH = 0.98
# The zone is widened by this much on each side after trimming, and
# never allowed to shrink below this size per axis (a degenerate zone
# would make the cursor jumpy).
ZONE_MARGIN = 0.02
ZONE_MIN_SIZE = 0.2

# --- Mouse / system control ---
MOUSE_SENSITIVITY = 0.3  # exponential smoothing factor for cursor motion
SCROLL_SPEED = 10
ZOOM_SPEED = 5
# Minimum seconds between repeated discrete actions (click, back, ...).
ACTION_COOLDOWN_SECONDS = 0.2

# HandoraX

*Control your PC with the power of your hands!*

HandoraX turns your webcam into a gesture-based controller: navigate,
click, drag, scroll, and zoom using intuitive hand gestures. Built with
Python, OpenCV, and MediaPipe (Tasks HandLandmarker).

## Features

- **Cursor Control**: Move your cursor with an open hand.
- **Click & Hold**: Tap index to thumb for a click, hold (>0.3s) to drag.
- **Multi-Gesture Support**: Right-click, double-click, drag, select,
  scroll, zoom, browser back/forward — all with simple hand movements.
- **Pause gesture**: Thumbs-up (or the `p` key) pauses and resumes
  control, so a hand in frame doesn't move your mouse when you don't
  want it to.
- **Customizable**: All thresholds live in `config.py`.
- **Real-Time**: Smooth tracking powered by MediaPipe, with cursor
  smoothing and per-frame movement (no more 5 Hz choppiness).

## Installation

Prerequisites: Python 3.10+ and a webcam.

```bash
git clone https://github.com/Altonomega/HandoraX.git
cd HandoraX
pip install -r requirements.txt
python main.py
```

On first run the hand landmarker model (~8 MB) is downloaded
automatically from Google's official model store into `models/` (not
committed to the repo).

Verify your setup without a camera:

```bash
python main.py --check
```

## Calibration: a comfortable range covers the whole screen

Mapping the whole camera frame 1:1 onto the screen is tiring: your
hand has to cross the entire frame (and often leaves it) to cross the
screen. HandoraX calibrates instead:

1. **Phase 1 (3s):** hold your hand open — this finds the palm centre.
2. **Phase 2 (5s):** move your hand around your *comfortable* range —
   as far as feels relaxed, no further. That range is stretched onto
   the full screen, so a small, relaxed movement reaches every pixel
   and your hand never needs to leave the frame. Positions outside the
   range clamp to the screen edge instead of getting lost.

The result is saved to `calibration.json` and reused on later runs.
Recalibrate any time with the `c` key or `python main.py
--recalibrate` — for example if you move the camera or switch chairs.

## Supported Gestures

| Gesture | Action | Description |
| --- | --- | --- |
| **Open Hand** | Move Cursor | Move your hand to control the cursor. |
| **Index to Thumb** | Click / Hold | Quick tap for click, hold (>0.3s) to drag. |
| **Two Fingers to Thumb** | Right Click | Pinch index and middle fingers to thumb. |
| **Three Fingers to Thumb** | Double Click | Pinch index, middle, and ring to thumb. |
| **Full Fist** | Drag | Close your fist to drag items. |
| **Fist with Index Out** | Select Text | Fist with only index extended. |
| **Middle to Thumb** | Scroll | Pinch middle finger to thumb and move up/down. |
| **Two Hands** | Zoom | Adjust distance between hands to zoom in/out. |
| **Pinky to Thumb** | Go Back | Pinch pinky to thumb (e.g., browser back). |
| **Ring to Thumb** | Go Forward | Pinch ring finger to thumb (e.g., browser forward). |
| **Thumbs-Up** | Pause / Resume | Toggle control on and off. |

Keys while running: `q` quits, `p` pauses/resumes, `c`
recalibrates, `--debug` logs every recognized gesture.

## How It Works

Three stages, kept in separate modules:

- **Hand Detection** (`hand_detector.py`): MediaPipe Tasks
  HandLandmarker tracks 21 landmarks per hand in real time.
- **Gesture Recognition** (`gesture_recognizer.py`): Pure logic over
  landmark geometry — no camera, no MediaPipe, no GUI. Pinch thresholds
  scale with the hand's own size (so distance from the camera doesn't
  change behaviour), finger extension is orientation-independent, and
  gestures are checked most-specific-first so a fist can't read as a
  click and a three-finger pinch can't be swallowed by the two-finger
  check.
- **System Control** (`system_controller.py`): Translates gestures
  into mouse/keyboard actions via PyAutoGUI, with smoothing for
  continuous motion and a cooldown for discrete actions.

## Configuration

Fine-tune HandoraX by editing `config.py`:

- **`MOUSE_SENSITIVITY`**: Cursor smoothing factor. *Default: 0.3*
- **`SCROLL_SPEED`**: Scroll sensitivity. *Default: 10*
- **`ZOOM_SPEED`**: Zoom sensitivity. *Default: 5*
- **`PINCH_RATIO`**: Pinch threshold as a fraction of hand size.
  *Default: 0.35*
- **`CLICK_HOLD_SECONDS`**: Pinch duration that becomes a hold.
  *Default: 0.3*
- **`PAUSE_DEBOUNCE_SECONDS`**: Minimum gap between pause toggles.
  *Default: 1.0*
- **`RANGE_CALIBRATION_SECONDS`**: Phase-2 calibration duration.
  *Default: 5.0*
- **`ZONE_MIN_SIZE`**: Smallest allowed active-zone size per axis.
  *Default: 0.2*

## Tests

The gesture logic is covered by unit tests with synthetic landmarks —
no camera or display needed:

```bash
python3 -m unittest discover -s tests -v
```

## Troubleshooting

- **Webcam Not Responding?**
  Double-check that your webcam is plugged in and free from other apps
  stealing its focus. Try another index: `python main.py --camera 1`.
- **Gestures Not Registering?**
  Make sure your lighting is bright and your hand stays in the frame
  during calibration.
- **Model download fails?**
  Check your connection, or download
  `hand_landmarker.task` manually into `models/` (see `MODEL_URL` in
  `config.py`).
- **Experiencing Lag?**
  Lower your webcam resolution or fine-tune the sensitivity settings
  in `config.py` for a smoother ride.

## Changelog

### Calibration & optimization (2026-10-25)

- Two-phase calibration: the user's comfortable hand range is measured
  and mapped onto the full screen, so relaxed hand movements reach
  every pixel before the hand leaves the frame. Outlier trimming
  (2nd/98th percentiles), a margin, and a minimum zone size keep the
  mapping stable; positions outside the zone clamp to the edge.
- Calibration persists in `calibration.json`; `c` or `--recalibrate`
  redo it.
- 8 new unit tests for zone mapping, clamping, trimming, and
  persistence (23 total, all green).

### Revival (2026-10-25)

- Migrated from the deprecated `mp.solutions.hands` API to the
  MediaPipe Tasks HandLandmarker, with automatic model download.
- Fixed double-click being unreachable (two-finger check shadowed the
  three-finger check) and fists misreading as clicks.
- Pinch thresholds now scale with hand size; finger detection no
  longer assumes an upright hand.
- Clicks fire exactly once per tap (previously a held pinch could
  repeat-click); cursor movement is no longer gated to 5 Hz.
- Added the thumbs-up pause gesture and `p` key; removed the
  constructor's surprise mouse movement and the unused `eye_tracker`
  module.
- Pinned dependency ranges; added 15 unit tests for the gesture logic.

## License

HandoraX is released under the [MIT License](LICENSE). Feel free to
use and adapt it, just keep the credit where it's due!

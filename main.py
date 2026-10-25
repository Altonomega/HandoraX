"""HandoraX entry point: webcam -> HandLandmarker -> gestures -> mouse.

Usage:
    python main.py [--debug] [--model PATH] [--camera INDEX]
    python main.py --check     # verify mediapipe + model without a camera

Calibration runs on first start and is saved to calibration.json:
  phase 1 (open hand, 3s) finds the palm centre;
  phase 2 (free movement, 5s) measures your comfortable hand range and
  maps THAT zone onto the full screen -- a small, relaxed movement
  covers every pixel, and your hand never needs to leave the frame.
A saved calibration is reused on later runs; --recalibrate or the 'c'
key runs it again.

Keys while running:
    q  quit
    p  pause / resume (same as the thumbs-up gesture)
    c  recalibrate
"""

import argparse
import json
import os
import sys
import time

from config import (
    CALIBRATION_FILE,
    CALIBRATION_SECONDS,
    RANGE_CALIBRATION_SECONDS,
)
from gesture_recognizer import GestureRecognizer
from hand_detector import HandDetector
from system_controller import SystemController

GESTURE_HELP = (
    "Move (open hand), Click/Hold (index to thumb), Right Click "
    "(index+middle to thumb), Double Click (index+middle+ring to thumb), "
    "Drag (fist), Select (fist + index out), Scroll (middle to thumb), "
    "Zoom (two hands), Back (pinky to thumb), Forward (ring to thumb), "
    "Pause (thumbs-up or 'p')"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HandoraX gesture control")
    parser.add_argument("--debug", action="store_true",
                        help="log every recognized gesture")
    parser.add_argument("--model", default=None,
                        help="path to hand_landmarker.task "
                             "(default: models/hand_landmarker.task)")
    parser.add_argument("--camera", type=int, default=0,
                        help="webcam index (default: 0)")
    parser.add_argument("--check", action="store_true",
                        help="verify mediapipe import and model file, then exit")
    parser.add_argument("--recalibrate", action="store_true",
                        help="ignore any saved calibration and recalibrate")
    return parser.parse_args(argv)


def load_calibration(recognizer, path=CALIBRATION_FILE):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            recognizer.apply_calibration(json.load(fh))
    except (OSError, ValueError) as exc:
        print(f"Ignoring unreadable calibration {path}: {exc}")
        return False
    print(f"Loaded saved calibration from {path} "
          f"(zone {tuple(round(v, 2) for v in recognizer.zone)}). "
          "Press 'c' to recalibrate.")
    return True


def save_calibration(recognizer, path=CALIBRATION_FILE):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(recognizer.calibration_dict(), fh, indent=2)
    print(f"Calibration saved to {path}.")


def run_calibration(cap, hand_detector, recognizer):
    """Two-phase calibration. Returns False if the user quit with 'q'."""
    import cv2

    print("Calibration phase 1: hold your hand open for "
          f"{CALIBRATION_SECONDS:.0f} seconds.")
    recognizer.reset_calibration()
    start_time = time.time()
    while time.time() - start_time < CALIBRATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame during calibration.")
            break
        hands = hand_detector.detect(frame)
        if hands:
            recognizer.calibrate(hands)
        cv2.imshow("HandoraX - Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    print("Calibration phase 2: move your hand around your COMFORTABLE "
          "range -- as far as feels relaxed, no further -- for "
          f"{RANGE_CALIBRATION_SECONDS:.0f} seconds. This range will "
          "cover the whole screen.")
    recognizer.begin_range_calibration()
    start_time = time.time()
    while time.time() - start_time < RANGE_CALIBRATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame during calibration.")
            break
        hands = hand_detector.detect(frame)
        if hands:
            recognizer.add_range_sample(hands)
        cv2.imshow("HandoraX - Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    if recognizer.finish_range_calibration():
        save_calibration(recognizer)
    else:
        print("Range calibration had too few samples; keeping the "
              "previous zone.")
    cv2.destroyWindow("HandoraX - Calibration")
    print("Calibration done!")
    return True


def main(argv=None):
    args = parse_args(argv)

    if args.check:
        try:
            HandDetector(model_path=args.model)
        except RuntimeError as exc:
            print(f"Check failed: {exc}")
            return 1
        print("Check OK: mediapipe imports and the hand landmarker model "
              "is present.")
        return 0

    import cv2

    try:
        hand_detector = HandDetector(model_path=args.model)
    except RuntimeError as exc:
        print(f"Cannot start: {exc}")
        return 1
    recognizer = GestureRecognizer(debug=args.debug)
    controller = SystemController(debug=args.debug)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}. Is a webcam connected?")
        return 1

    try:
        calibrated = (
            not args.recalibrate and load_calibration(recognizer)
        )
        if not calibrated:
            if not run_calibration(cap, hand_detector, recognizer):
                return 0

        print(f"Starting! {GESTURE_HELP}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame!")
                break

            hands = hand_detector.detect(frame)
            gesture_data = recognizer.recognize(hands or [])
            controller.execute(gesture_data)

            label = gesture_data["gesture"]
            if recognizer.paused:
                label += " (PAUSED - thumbs-up or 'p' to resume)"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("HandoraX", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = recognizer.toggle_pause()
                print(f"Paused: {paused}")
            if key == ord("c"):
                print("Recalibrating...")
                if not run_calibration(cap, hand_detector, recognizer):
                    break
                print(f"Resuming! {GESTURE_HELP}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

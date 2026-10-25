"""HandoraX entry point: webcam -> HandLandmarker -> gestures -> mouse.

Usage:
    python main.py [--debug] [--model PATH] [--camera INDEX]
    python main.py --check     # verify mediapipe + model without a camera

Keys while running:
    q  quit
    p  pause / resume (same as the thumbs-up gesture)
"""

import argparse
import sys
import time

from config import CALIBRATION_SECONDS
from gesture_recognizer import GestureRecognizer
from hand_detector import HandDetector, ensure_model
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
    return parser.parse_args(argv)


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

    print("Calibration... hold your hand open for "
          f"{CALIBRATION_SECONDS:.0f} seconds.")
    start_time = time.time()
    try:
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
                return 0

        print(f"Calibration done! {GESTURE_HELP}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame!")
                break

            hands = hand_detector.detect(frame)
            if hands:
                gesture_data = recognizer.recognize(hands)
            else:
                gesture_data = recognizer.recognize([])
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
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_detector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

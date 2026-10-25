"""Gesture recognition from MediaPipe hand landmarks.

Pure logic: no camera, no MediaPipe, no pyautogui here, so the whole
module is unit-testable with synthetic landmarks.

A "hand" is a sequence of 21 points with .x/.y attributes (exactly what
both the legacy MediaPipe API and the Tasks HandLandmarker return).

Recognition precedence is deliberate, most specific first:
  1. thumbs-up            -> pause toggle (state lives here)
  2. full fist            -> drag
  3. fist + index out     -> select
  4. three-finger pinch   -> double click
  5. two-finger pinch     -> right click
  6. index pinch          -> click / hold state machine
  7. middle pinch         -> scroll
  8. two hands            -> zoom
  9. pinky pinch          -> back
 10. ring pinch           -> forward
 11. index extended       -> move

This order fixes two real bugs in the original code:
  - Double-click was unreachable: the two-finger check ran before the
    three-finger check and returned first.
  - A curled fist read as a click: curled fingertips sit near the thumb,
    so the pinch checks fired before the fist checks. Fists now win.
"""

import time

from config import (
    CLICK_HOLD_SECONDS,
    FINGER_EXTENDED_RATIO,
    PAUSE_DEBOUNCE_SECONDS,
    PINCH_RATIO,
)

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18
MIDDLE_MCP = 9
_PALM_POINTS = (0, 5, 9, 17)
_FINGERS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _palm(hand):
    n = len(_PALM_POINTS)
    return (
        sum(hand[i].x for i in _PALM_POINTS) / n,
        sum(hand[i].y for i in _PALM_POINTS) / n,
    )


class GestureRecognizer:
    def __init__(self, debug=False, time_fn=None):
        self.debug = debug
        self._now = time_fn or time.time
        self.calibrated = False
        self.base_x, self.base_y = 0.5, 0.5
        self._cal_samples = []
        self._pinch_start = None
        self.holding = False
        self.paused = False
        self._last_pause_toggle = 0.0
        self._prev_thumbs_up = False

    # -- logging ------------------------------------------------------
    def _log(self, msg):
        if self.debug:
            print(msg)

    # -- calibration --------------------------------------------------
    def calibrate(self, hands):
        """Accumulate one palm sample; the base is the running mean."""
        if hands:
            palm_x, palm_y = _palm(hands[0])
            self._cal_samples.append((palm_x, palm_y))
            n = len(self._cal_samples)
            self.base_x = sum(p[0] for p in self._cal_samples) / n
            self.base_y = sum(p[1] for p in self._cal_samples) / n
            self.calibrated = True
            self._log(f"Calibrated at: ({self.base_x:.2f}, {self.base_y:.2f})")

    def reset_calibration(self):
        self._cal_samples = []
        self.base_x, self.base_y = 0.5, 0.5
        self.calibrated = False

    # -- pause --------------------------------------------------------
    def toggle_pause(self):
        self.paused = not self.paused
        self._last_pause_toggle = self._now()
        return self.paused

    # -- helpers ------------------------------------------------------
    def _extended(self, hand, tip_idx, pip_idx):
        wrist = hand[WRIST]
        return _dist(hand[tip_idx], wrist) > (
            FINGER_EXTENDED_RATIO * _dist(hand[pip_idx], wrist)
        )

    def _pinched(self, hand, tip_idx, hand_size):
        return _dist(hand[tip_idx], hand[THUMB_TIP]) < PINCH_RATIO * hand_size

    def _result(self, gesture, position, active, **extra):
        out = {"gesture": gesture, "position": position, "active": active}
        out.update(extra)
        return out

    # -- main ---------------------------------------------------------
    def recognize(self, hands):
        if not hands:
            if self.holding:
                self.holding = False
                self._pinch_start = None
                return self._result(
                    "stop_hold", (self.base_x, self.base_y), False
                )
            return self._result("stop", (self.base_x, self.base_y), False)

        hand = hands[0]
        palm = _palm(hand)
        hand_size = _dist(hand[WRIST], hand[MIDDLE_MCP])
        if hand_size <= 0:
            return self._result("stop", palm, False)

        extended = {
            name: self._extended(hand, tip, pip)
            for name, (tip, pip) in (
                ("index", (INDEX_TIP, INDEX_PIP)),
                ("middle", (MIDDLE_TIP, MIDDLE_PIP)),
                ("ring", (RING_TIP, RING_PIP)),
                ("pinky", (PINKY_TIP, PINKY_PIP)),
                ("thumb", (THUMB_TIP, THUMB_IP)),
            )
        }
        pinched = {
            name: self._pinched(hand, tip, hand_size)
            for name, tip in (
                ("index", INDEX_TIP),
                ("middle", MIDDLE_TIP),
                ("ring", RING_TIP),
                ("pinky", PINKY_TIP),
            )
        }

        # 1. Thumbs-up toggles pause, on the rising edge only, debounced.
        # It must precede the fist check: a thumbs-up hand has four curled
        # fingers and would otherwise read as a fist.
        thumbs_up = extended["thumb"] and not (
            extended["index"]
            or extended["middle"]
            or extended["ring"]
            or extended["pinky"]
        )
        now = self._now()
        if (
            thumbs_up
            and not self._prev_thumbs_up
            and now - self._last_pause_toggle >= PAUSE_DEBOUNCE_SECONDS
        ):
            paused = self.toggle_pause()
            self._prev_thumbs_up = True
            self._log(f"Pause toggled: paused={paused}")
            return self._result("pause_toggle", palm, False, paused=paused)
        self._prev_thumbs_up = thumbs_up
        if self.paused:
            return self._result("paused", palm, False)

        # 2. Full fist -> drag.
        if not any(
            extended[f] for f in ("index", "middle", "ring", "pinky")
        ):
            self._pinch_start = None
            self._log("Full fist - Drag!")
            return self._result("drag", palm, True)

        # 3. Fist with index out -> select. A pinch on any other finger
        # takes precedence: "index out + middle pinched" is scroll, not
        # select, and the same holds for the pinky/ring pinches below.
        if (
            extended["index"]
            and not (
                extended["middle"] or extended["ring"] or extended["pinky"]
            )
            and not (
                pinched["middle"] or pinched["ring"] or pinched["pinky"]
            )
        ):
            self._pinch_start = None
            self._log("Fist with index out - Select!")
            return self._result("select", palm, True)

        # 4. Three-finger pinch -> double click. Checked BEFORE the
        # two-finger check: the old order made this unreachable.
        if pinched["index"] and pinched["middle"] and pinched["ring"]:
            self._pinch_start = None
            self.holding = False
            self._log("Three fingers to thumb - Double Click!")
            return self._result("double_click", palm, True)

        # 5. Two-finger pinch -> right click.
        if pinched["index"] and pinched["middle"]:
            self._pinch_start = None
            self.holding = False
            self._log("Two fingers to thumb - Right Click!")
            return self._result("right_click", palm, True)

        # 6. Index pinch -> click / hold state machine. "click" fires
        # exactly once, on the pinch-start frame; later frames report
        # "pinch" (a no-op for the controller) until the hold threshold,
        # so a single tap can never repeat-click.
        if pinched["index"]:
            if self.holding:
                self._log("Index to thumb - Holding continued!")
                return self._result("hold", palm, True)
            if self._pinch_start is None:
                self._pinch_start = now
                self._log("Index to thumb - Click!")
                return self._result("click", palm, True)
            if now - self._pinch_start > CLICK_HOLD_SECONDS:
                self.holding = True
                self._pinch_start = None
                self._log("Index to thumb - Holding!")
                return self._result("hold", palm, True)
            return self._result("pinch", palm, True)

        # Pinch released: a hold ends with stop_hold, a quick tap needs
        # no further event (its click already fired on the start frame).
        if self.holding:
            self.holding = False
            self._log("Hold released - Stop!")
            return self._result("stop_hold", palm, False)
        self._pinch_start = None

        # 7. Middle pinch -> scroll.
        if pinched["middle"]:
            delta_y = palm[1] - self.base_y
            self._log(f"Middle to thumb - Scroll! Delta Y: {delta_y:.2f}")
            return self._result(
                "scroll", palm, True, delta_y=delta_y
            )

        # 8. Two hands -> zoom.
        if len(hands) > 1:
            second_palm = _palm(hands[1])
            dist_hands = (
                (palm[0] - second_palm[0]) ** 2
                + (palm[1] - second_palm[1]) ** 2
            ) ** 0.5
            self._log(f"Zoom gesture - Distance: {dist_hands:.2f}")
            return self._result("zoom", palm, True, dist=dist_hands)

        # 9. Pinky pinch -> back.
        if pinched["pinky"]:
            self._log("Pinky to thumb - Back!")
            return self._result("back", palm, True)

        # 10. Ring pinch -> forward.
        if pinched["ring"]:
            self._log("Ring to thumb - Forward!")
            return self._result("forward", palm, True)

        # 11. Index extended -> move.
        if extended["index"]:
            self._log("Hand open - Move!")
            return self._result("move", palm, True)

        self._log("Stop!")
        return self._result("stop", palm, False)

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

Positions in returned dicts are SCREEN-normalized coordinates in
[0, 1] (mirrored horizontally, as a mirror feels natural): the
recognizer maps the raw palm position through the calibrated active
zone (see map_position), so a small comfortable hand range covers the
whole screen. The controller multiplies by the screen size; it must
not mirror again.
"""

import time

from config import (
    CLICK_HOLD_SECONDS,
    FINGER_EXTENDED_RATIO,
    PAUSE_DEBOUNCE_SECONDS,
    PINCH_RATIO,
    ZONE_MARGIN,
    ZONE_MIN_SIZE,
    ZONE_PERCENTILE_HIGH,
    ZONE_PERCENTILE_LOW,
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


def _percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    idx = int(p * (len(sorted_values) - 1))
    return sorted_values[max(0, min(len(sorted_values) - 1, idx))]


def _clamp01(v):
    return min(1.0, max(0.0, v))


class GestureRecognizer:
    def __init__(self, debug=False, time_fn=None):
        self.debug = debug
        self._now = time_fn or time.time
        self.calibrated = False
        self.base_x, self.base_y = 0.5, 0.5
        self._cal_samples = []
        # Active zone as (x_min, x_max, y_min, y_max) in camera
        # coordinates. Default is the whole frame: the old 1:1 behaviour.
        self.zone = (0.0, 1.0, 0.0, 1.0)
        self._range_samples = []
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

    # -- active zone --------------------------------------------------
    def begin_range_calibration(self):
        """Start collecting palm samples for the comfortable range."""
        self._range_samples = []

    def add_range_sample(self, hands):
        if hands:
            self._range_samples.append(_palm(hands[0]))

    def finish_range_calibration(self):
        """Turn collected samples into the active zone.

        The 2nd/98th percentiles trim stray frames, a margin is added,
        and the zone is never allowed below ZONE_MIN_SIZE per axis.
        Returns True when a zone was computed, False when too few
        samples arrived (the previous zone is kept).
        """
        samples = self._range_samples
        if len(samples) < 10:
            self._log("Too few range samples; keeping previous zone.")
            return False
        xs = sorted(p[0] for p in samples)
        ys = sorted(p[1] for p in samples)
        x0 = _percentile(xs, ZONE_PERCENTILE_LOW) - ZONE_MARGIN
        x1 = _percentile(xs, ZONE_PERCENTILE_HIGH) + ZONE_MARGIN
        y0 = _percentile(ys, ZONE_PERCENTILE_LOW) - ZONE_MARGIN
        y1 = _percentile(ys, ZONE_PERCENTILE_HIGH) + ZONE_MARGIN
        # Enforce a minimum size, expanding around the centre.
        for axis in (0, 1):
            lo, hi = (x0, x1) if axis == 0 else (y0, y1)
            if hi - lo < ZONE_MIN_SIZE:
                mid = (lo + hi) / 2
                lo, hi = mid - ZONE_MIN_SIZE / 2, mid + ZONE_MIN_SIZE / 2
            lo, hi = max(0.0, lo), min(1.0, hi)
            if axis == 0:
                x0, x1 = lo, hi
            else:
                y0, y1 = lo, hi
        self.zone = (x0, x1, y0, y1)
        self._log(
            f"Active zone: x [{x0:.2f}, {x1:.2f}] y [{y0:.2f}, {y1:.2f}]"
        )
        return True

    def map_position(self, palm):
        """Map a raw camera palm position onto screen coordinates.

        The position inside the active zone is stretched to [0, 1] and
        clamped, so the whole zone covers the whole screen, and the
        x-axis is mirrored (moving your hand right moves the cursor
        right, as in a mirror).
        """
        x0, x1, y0, y1 = self.zone
        nx = (palm[0] - x0) / (x1 - x0) if x1 > x0 else 0.5
        ny = (palm[1] - y0) / (y1 - y0) if y1 > y0 else 0.5
        return (1.0 - _clamp01(nx), _clamp01(ny))

    def calibration_dict(self):
        return {
            "zone": list(self.zone),
            "base": [self.base_x, self.base_y],
        }

    def apply_calibration(self, data):
        zone = data.get("zone")
        base = data.get("base")
        if zone and len(zone) == 4:
            self.zone = tuple(float(v) for v in zone)
        if base and len(base) == 2:
            self.base_x, self.base_y = float(base[0]), float(base[1])
            self.calibrated = True

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
            base_pos = self.map_position((self.base_x, self.base_y))
            if self.holding:
                self.holding = False
                self._pinch_start = None
                return self._result("stop_hold", base_pos, False)
            return self._result("stop", base_pos, False)

        hand = hands[0]
        palm = _palm(hand)
        pos = self.map_position(palm)
        hand_size = _dist(hand[WRIST], hand[MIDDLE_MCP])
        if hand_size <= 0:
            return self._result("stop", pos, False)

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
            return self._result("pause_toggle", pos, False, paused=paused)
        self._prev_thumbs_up = thumbs_up
        if self.paused:
            return self._result("paused", pos, False)

        # 2. Full fist -> drag.
        if not any(
            extended[f] for f in ("index", "middle", "ring", "pinky")
        ):
            self._pinch_start = None
            self._log("Full fist - Drag!")
            return self._result("drag", pos, True)

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
            return self._result("select", pos, True)

        # 4. Three-finger pinch -> double click. Checked BEFORE the
        # two-finger check: the old order made this unreachable.
        if pinched["index"] and pinched["middle"] and pinched["ring"]:
            self._pinch_start = None
            self.holding = False
            self._log("Three fingers to thumb - Double Click!")
            return self._result("double_click", pos, True)

        # 5. Two-finger pinch -> right click.
        if pinched["index"] and pinched["middle"]:
            self._pinch_start = None
            self.holding = False
            self._log("Two fingers to thumb - Right Click!")
            return self._result("right_click", pos, True)

        # 6. Index pinch -> click / hold state machine. "click" fires
        # exactly once, on the pinch-start frame; later frames report
        # "pinch" (a no-op for the controller) until the hold threshold,
        # so a single tap can never repeat-click.
        if pinched["index"]:
            if self.holding:
                self._log("Index to thumb - Holding continued!")
                return self._result("hold", pos, True)
            if self._pinch_start is None:
                self._pinch_start = now
                self._log("Index to thumb - Click!")
                return self._result("click", pos, True)
            if now - self._pinch_start > CLICK_HOLD_SECONDS:
                self.holding = True
                self._pinch_start = None
                self._log("Index to thumb - Holding!")
                return self._result("hold", pos, True)
            return self._result("pinch", pos, True)

        # Pinch released: a hold ends with stop_hold, a quick tap needs
        # no further event (its click already fired on the start frame).
        if self.holding:
            self.holding = False
            self._log("Hold released - Stop!")
            return self._result("stop_hold", pos, False)
        self._pinch_start = None

        # 7. Middle pinch -> scroll.
        if pinched["middle"]:
            delta_y = palm[1] - self.base_y
            self._log(f"Middle to thumb - Scroll! Delta Y: {delta_y:.2f}")
            return self._result(
                "scroll", pos, True, delta_y=delta_y
            )

        # 8. Two hands -> zoom.
        if len(hands) > 1:
            second_palm = _palm(hands[1])
            dist_hands = (
                (palm[0] - second_palm[0]) ** 2
                + (palm[1] - second_palm[1]) ** 2
            ) ** 0.5
            self._log(f"Zoom gesture - Distance: {dist_hands:.2f}")
            return self._result("zoom", pos, True, dist=dist_hands)

        # 9. Pinky pinch -> back.
        if pinched["pinky"]:
            self._log("Pinky to thumb - Back!")
            return self._result("back", pos, True)

        # 10. Ring pinch -> forward.
        if pinched["ring"]:
            self._log("Ring to thumb - Forward!")
            return self._result("forward", pos, True)

        # 11. Index extended -> move.
        if extended["index"]:
            self._log("Hand open - Move!")
            return self._result("move", pos, True)

        self._log("Stop!")
        return self._result("stop", pos, False)

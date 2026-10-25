"""Unit tests for gesture_recognizer, using synthetic landmarks.

No camera, no MediaPipe, no display needed: a hand is a list of 21
points with .x/.y, built by make_hand() below. Run with:

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gesture_recognizer import GestureRecognizer  # noqa: E402

# Finger geometry: wrist at (0.5, 0.8), fingers pointing up.
_WRIST = (0.5, 0.8)
# finger -> (mcp, pip_offset, tip_offset) for the extended pose; the
# curled pose folds the tip back toward the palm.
_FINGERS = {
    "index": ((0.45, 0.60), (0.0, -0.08), (0.0, -0.16)),
    "middle": ((0.50, 0.60), (0.0, -0.08), (0.0, -0.16)),
    "ring": ((0.55, 0.60), (0.0, -0.08), (0.0, -0.16)),
    "pinky": ((0.60, 0.62), (0.0, -0.07), (0.0, -0.14)),
}
# landmark indices per finger: (mcp, pip, dip, tip)
_IDX = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_THUMB = (1, 2, 3, 4)
# Curled thumb tip rests across the palm, close to the wrist: far enough
# from the index knuckle line that the extension test reads it as curled
# (a tip parked far out would read as an extended thumb).
_THUMB_TIP = (0.44, 0.66)


def _pt(xy):
    return SimpleNamespace(x=xy[0], y=xy[1])


def make_hand(extended=(), pinched=(), thumb_extended=False,
              tips_on_thumb=False):
    """Build a 21-landmark hand.

    extended: fingers held out. pinched: fingertips placed on the thumb
    tip. Everything else is curled. tips_on_thumb: curl all fingertips
    onto the thumb tip (a fist whose tips touch the thumb) -- the case
    the old code misread as a click.
    """
    pts = [None] * 21
    pts[0] = _pt(_WRIST)
    # Thumb joints.
    if thumb_extended:
        thumb_pts = [(0.42, 0.70), (0.38, 0.65), (0.34, 0.55)]
    else:
        thumb_pts = [(0.42, 0.70), (0.41, 0.68), _THUMB_TIP]
    for idx, xy in zip(_THUMB[1:], thumb_pts):
        pts[idx] = _pt(xy)
    pts[_THUMB[0]] = _pt((0.44, 0.72))
    for name, (mcp, pip_off, tip_off) in _FINGERS.items():
        mcp_i, pip_i, dip_i, tip_i = _IDX[name]
        pts[mcp_i] = _pt(mcp)
        if name in pinched:
            pts[pip_i] = _pt((mcp[0] + pip_off[0], mcp[1] + pip_off[1]))
            pts[dip_i] = _pt((mcp[0] + pip_off[0], mcp[1] + pip_off[1]))
            pts[tip_i] = _pt((0.41, 0.62))  # on the thumb tip
        elif name in extended:
            pts[pip_i] = _pt((mcp[0] + pip_off[0], mcp[1] + pip_off[1]))
            pts[dip_i] = _pt((mcp[0] + pip_off[0] * 1.5,
                              mcp[1] + pip_off[1] * 1.5))
            pts[tip_i] = _pt((mcp[0] + tip_off[0], mcp[1] + tip_off[1]))
        elif tips_on_thumb:
            pts[pip_i] = _pt((mcp[0] + pip_off[0], mcp[1] + pip_off[1]))
            pts[dip_i] = _pt((mcp[0], mcp[1] - 0.02))
            pts[tip_i] = _pt((0.42, 0.64))  # touching the thumb tip
        else:
            pts[pip_i] = _pt((mcp[0] + pip_off[0], mcp[1] + pip_off[1]))
            pts[dip_i] = _pt((mcp[0] + 0.01, mcp[1] - 0.02))
            pts[tip_i] = _pt((mcp[0] + 0.02, mcp[1] - 0.02))
    assert all(p is not None for p in pts)
    return pts


def rotate_hand(hand, cx=0.5, cy=0.5):
    """Rotate a hand 90 degrees around (cx, cy): distances preserved,
    so extension/pinch results must not change."""
    out = []
    for p in hand:
        out.append(_pt((cx - (p.y - cy), cy + (p.x - cx))))
    return out


class FakeClock:
    def __init__(self, start=100.0):
        self.t = start

    def __call__(self):
        return self.t


class TestGestures(unittest.TestCase):
    def recognizer(self):
        self.clock = FakeClock()
        return GestureRecognizer(time_fn=self.clock)

    def test_three_finger_pinch_is_double_click(self):
        """Regression: the old order made double-click unreachable."""
        rec = self.recognizer()
        # Natural pinch pose: the non-pinching finger stays open, so the
        # hand is not a fist.
        hand = make_hand(pinched=("index", "middle", "ring"),
                         extended=("pinky",))
        self.assertEqual(rec.recognize([hand])["gesture"], "double_click")

    def test_two_finger_pinch_is_right_click(self):
        rec = self.recognizer()
        hand = make_hand(pinched=("index", "middle"),
                         extended=("ring", "pinky"))
        self.assertEqual(rec.recognize([hand])["gesture"], "right_click")

    def test_fist_is_drag_not_click(self):
        rec = self.recognizer()
        hand = make_hand()
        self.assertEqual(rec.recognize([hand])["gesture"], "drag")

    def test_fist_with_tips_on_thumb_is_still_drag(self):
        """Curled fingertips sit near the thumb; that must not read
        as a pinch. This is the second precedence bug."""
        rec = self.recognizer()
        hand = make_hand(tips_on_thumb=True)
        self.assertEqual(rec.recognize([hand])["gesture"], "drag")

    def test_fist_with_index_out_is_select(self):
        rec = self.recognizer()
        hand = make_hand(extended=("index",))
        self.assertEqual(rec.recognize([hand])["gesture"], "select")

    def test_open_hand_moves(self):
        rec = self.recognizer()
        hand = make_hand(extended=("index", "middle", "ring", "pinky"))
        self.assertEqual(rec.recognize([hand])["gesture"], "move")

    def test_fist_detection_is_orientation_independent(self):
        """The old tip-above-wrist test broke on a tilted hand."""
        rec = self.recognizer()
        hand = rotate_hand(make_hand())
        self.assertEqual(rec.recognize([hand])["gesture"], "drag")

    def test_click_fires_once_then_hold(self):
        rec = self.recognizer()
        hand = make_hand(pinched=("index",),
                         extended=("middle", "ring", "pinky"))
        self.assertEqual(rec.recognize([hand])["gesture"], "click")
        # Still pinched, under the hold threshold: no repeat click.
        self.clock.t += 0.1
        self.assertEqual(rec.recognize([hand])["gesture"], "pinch")
        # Past the threshold: hold.
        self.clock.t += 0.3
        self.assertEqual(rec.recognize([hand])["gesture"], "hold")
        # Released: stop_hold.
        open_hand = make_hand(extended=("index", "middle", "ring", "pinky"))
        self.assertEqual(rec.recognize([open_hand])["gesture"], "stop_hold")

    def test_middle_pinch_scrolls(self):
        rec = self.recognizer()
        hand = make_hand(pinched=("middle",), extended=("index",))
        result = rec.recognize([hand])
        self.assertEqual(result["gesture"], "scroll")
        self.assertIn("delta_y", result)

    def test_two_hands_zoom(self):
        rec = self.recognizer()
        hand1 = make_hand(extended=("index", "middle", "ring", "pinky"))
        hand2 = make_hand(extended=("index", "middle", "ring", "pinky"))
        result = rec.recognize([hand1, hand2])
        self.assertEqual(result["gesture"], "zoom")
        self.assertIn("dist", result)

    def test_pinky_and_ring_pinches(self):
        rec = self.recognizer()
        hand = make_hand(pinched=("pinky",), extended=("index",))
        self.assertEqual(rec.recognize([hand])["gesture"], "back")
        hand = make_hand(pinched=("ring",), extended=("index",))
        self.assertEqual(rec.recognize([hand])["gesture"], "forward")

    def test_thumbs_up_toggles_pause(self):
        rec = self.recognizer()
        up = make_hand(thumb_extended=True)
        open_hand = make_hand(extended=("index", "middle", "ring", "pinky"))
        result = rec.recognize([up])
        self.assertEqual(result["gesture"], "pause_toggle")
        self.assertTrue(result["paused"])
        # Holding thumbs-up does not re-toggle on consecutive frames.
        self.assertEqual(rec.recognize([up])["gesture"], "paused")
        # While paused, an open hand does not move the mouse.
        self.assertEqual(rec.recognize([open_hand])["gesture"], "paused")
        # A fresh raise after the debounce unpauses.
        self.clock.t += 2.0
        result = rec.recognize([up])
        self.assertEqual(result["gesture"], "pause_toggle")
        self.assertFalse(result["paused"])
        self.assertEqual(rec.recognize([open_hand])["gesture"], "move")

    def test_no_hands_stops(self):
        rec = self.recognizer()
        self.assertEqual(rec.recognize([])["gesture"], "stop")
        self.assertEqual(rec.recognize(None)["gesture"], "stop")

    def test_calibration_averages_samples(self):
        rec = self.recognizer()
        hand1 = make_hand(extended=("index", "middle", "ring", "pinky"))
        hand2 = rotate_hand(hand1)
        rec.calibrate([hand1])
        rec.calibrate([hand2])
        # Base is the mean of the two palm positions, not the last one.
        palm1 = (
            sum(p.x for p in (hand1[0], hand1[5], hand1[9], hand1[17])) / 4,
            sum(p.y for p in (hand1[0], hand1[5], hand1[9], hand1[17])) / 4,
        )
        palm2 = (
            sum(p.x for p in (hand2[0], hand2[5], hand2[9], hand2[17])) / 4,
            sum(p.y for p in (hand2[0], hand2[5], hand2[9], hand2[17])) / 4,
        )
        self.assertAlmostEqual(rec.base_x, (palm1[0] + palm2[0]) / 2)
        self.assertAlmostEqual(rec.base_y, (palm1[1] + palm2[1]) / 2)
        self.assertTrue(rec.calibrated)

    def test_pinch_threshold_scales_with_hand_size(self):
        """Same 0.05 thumb gap: a pinch on a normal hand, not on a
        hand half the size. The old fixed 0.07 threshold could not do
        this."""
        rec = self.recognizer()
        hand = make_hand(extended=("middle", "ring", "pinky"))
        # Normal hand: put the index tip 0.05 from the thumb tip.
        hand[8] = _pt((0.45, 0.62))
        self.assertEqual(rec.recognize([hand])["gesture"], "click")
        # Half-size hand (all points pulled toward the wrist), same gap.
        small = []
        for p in hand:
            small.append(_pt((
                _WRIST[0] + (p.x - _WRIST[0]) * 0.5,
                _WRIST[1] + (p.y - _WRIST[1]) * 0.5,
            )))
        small[8] = _pt((0.45, 0.62))  # restore the 0.05 gap
        rec2 = self.recognizer()
        # Index is curled in this hand, so a non-pinch reads as drag
        # or stop -- but never as a click.
        self.assertNotEqual(rec2.recognize([small])["gesture"], "click")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the calibrated active zone.

The zone is what makes a small, comfortable hand movement cover the
whole screen: the calibrated hand range is stretched onto [0, 1] and
clamped, so the hand never needs to leave the camera frame. Run with:

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gesture_recognizer import GestureRecognizer  # noqa: E402


def palm_hand(x, y):
    """A 21-point hand whose palm centre is (x, y). Only the palm
    points (0, 5, 9, 17) matter for range calibration."""
    return [SimpleNamespace(x=x, y=y) for _ in range(21)]


class TestZoneMapping(unittest.TestCase):
    def test_default_zone_is_full_frame_mirrored(self):
        rec = GestureRecognizer()
        self.assertEqual(rec.map_position((0.0, 0.0)), (1.0, 0.0))
        self.assertEqual(rec.map_position((1.0, 1.0)), (0.0, 1.0))
        self.assertEqual(rec.map_position((0.5, 0.5)), (0.5, 0.5))

    def test_small_zone_covers_full_screen(self):
        """A hand range of 0.4..0.6 must still reach every screen edge."""
        rec = GestureRecognizer()
        rec.zone = (0.4, 0.6, 0.4, 0.6)
        # Left edge of the zone -> right edge of the screen (mirrored).
        self.assertEqual(rec.map_position((0.4, 0.4)), (1.0, 0.0))
        self.assertEqual(rec.map_position((0.6, 0.6)), (0.0, 1.0))
        self.assertEqual(rec.map_position((0.5, 0.5)), (0.5, 0.5))

    def test_positions_outside_zone_clamp_to_edge(self):
        rec = GestureRecognizer()
        rec.zone = (0.4, 0.6, 0.4, 0.6)
        self.assertEqual(rec.map_position((0.0, 0.0)), (1.0, 0.0))
        self.assertEqual(rec.map_position((1.0, 1.0)), (0.0, 1.0))

    def test_range_calibration_trims_outliers(self):
        """100 samples inside 0.4..0.6 plus one stray frame at each
        extreme: the zone must stay near the real range."""
        rec = GestureRecognizer()
        rec.begin_range_calibration()
        rec.add_range_sample([palm_hand(0.0, 0.0)])
        for i in range(100):
            t = 0.4 + 0.2 * (i / 99)
            rec.add_range_sample([palm_hand(t, t)])
        rec.add_range_sample([palm_hand(1.0, 1.0)])
        self.assertTrue(rec.finish_range_calibration())
        x0, x1, y0, y1 = rec.zone
        self.assertGreaterEqual(x0, 0.35)
        self.assertLessEqual(x1, 0.65)
        self.assertGreaterEqual(y0, 0.35)
        self.assertLessEqual(y1, 0.65)

    def test_range_calibration_enforces_minimum_size(self):
        """A tiny real range must not produce a jumpy micro-zone."""
        rec = GestureRecognizer()
        rec.begin_range_calibration()
        for i in range(20):
            t = 0.49 + 0.02 * (i / 19)
            rec.add_range_sample([palm_hand(t, t)])
        self.assertTrue(rec.finish_range_calibration())
        x0, x1, y0, y1 = rec.zone
        self.assertGreaterEqual(x1 - x0, 0.2 - 1e-9)
        self.assertGreaterEqual(y1 - y0, 0.2 - 1e-9)

    def test_range_calibration_needs_enough_samples(self):
        rec = GestureRecognizer()
        rec.begin_range_calibration()
        rec.add_range_sample([palm_hand(0.5, 0.5)])
        self.assertFalse(rec.finish_range_calibration())
        self.assertEqual(rec.zone, (0.0, 1.0, 0.0, 1.0))

    def test_calibration_persists_and_restores(self):
        rec = GestureRecognizer()
        rec.zone = (0.3, 0.7, 0.35, 0.65)
        rec.base_x, rec.base_y = 0.5, 0.55
        data = rec.calibration_dict()
        rec2 = GestureRecognizer()
        rec2.apply_calibration(data)
        self.assertEqual(rec2.zone, (0.3, 0.7, 0.35, 0.65))
        self.assertEqual((rec2.base_x, rec2.base_y), (0.5, 0.55))
        self.assertTrue(rec2.calibrated)

    def test_recognize_returns_mapped_position(self):
        """End to end through recognize: with a small zone, a palm at
        the zone centre maps to the screen centre."""
        sys.path.insert(0, os.path.dirname(__file__))
        from test_gesture_recognizer import make_hand
        rec = GestureRecognizer()
        rec.zone = (0.4, 0.6, 0.4, 0.6)
        hand = make_hand(extended=("index", "middle", "ring", "pinky"))
        result = rec.recognize([hand])
        self.assertEqual(result["gesture"], "move")
        sx, sy = result["position"]
        self.assertGreaterEqual(sx, 0.0)
        self.assertLessEqual(sx, 1.0)
        self.assertGreaterEqual(sy, 0.0)
        self.assertLessEqual(sy, 1.0)


if __name__ == "__main__":
    unittest.main()

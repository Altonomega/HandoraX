"""Translate recognized gestures into mouse/keyboard actions.

Changes vs. the original:
  - The constructor no longer moves the user's real mouse as a "test".
    That was a surprising side effect of merely starting the program.
  - Per-frame print spam is behind a debug flag.
  - The unused GESTURE_ACTIONS import is gone; dispatch is on the
    gesture names the recognizer actually emits.
  - "pause_toggle" / "paused" / "pinch" are explicitly no-ops here:
    pausing must never move the mouse, and a pending pinch must not
    click until it becomes a hold.
"""

import time

from config import (
    ACTION_COOLDOWN_SECONDS,
    MOUSE_SENSITIVITY,
    SCROLL_SPEED,
    ZOOM_SPEED,
)

# Gestures that intentionally do nothing at the system level.
_NOOP_GESTURES = ("pause_toggle", "paused", "pinch", "stop")
# Gestures that must run every frame (cursor tracking); the rest are
# discrete actions and stay behind the cooldown so one pose cannot
# machine-gun clicks. The old code gated everything at 5 Hz, which made
# the cursor visibly choppy.
_CONTINUOUS_GESTURES = ("move", "drag", "select", "hold")


class SystemController:
    def __init__(self, debug=False):
        import pyautogui

        self._pyautogui = pyautogui
        pyautogui.PAUSE = 0
        self.debug = debug
        self.prev_x = None
        self.prev_y = None
        self.smoothing = MOUSE_SENSITIVITY
        self.last_gesture = None
        self.last_action_time = 0
        self.cooldown = ACTION_COOLDOWN_SECONDS
        self.dragging = False
        self.selecting = False
        self.holding = False
        self.last_dist = None

    def _log(self, msg):
        if self.debug:
            print(msg)

    def _release_all(self):
        pyautogui = self._pyautogui
        if self.dragging:
            pyautogui.mouseUp()
            self.dragging = False
            self._log("Drag stopped!")
        if self.selecting:
            pyautogui.mouseUp()
            self.selecting = False
            self._log("Selection stopped!")
        if self.holding:
            pyautogui.mouseUp()
            self.holding = False
            self._log("Hold stopped!")

    def execute(self, gesture_data):
        pyautogui = self._pyautogui
        current_time = time.time()
        gesture = gesture_data["gesture"]
        palm_x, palm_y = gesture_data["position"]
        active = gesture_data["active"]

        self._log(
            f"Gesture: {gesture}, Active: {active}, "
            f"Position: ({palm_x:.2f}, {palm_y:.2f})"
        )

        if not active:
            self._release_all()
            return

        screen_width, screen_height = pyautogui.size()
        raw_x = (1 - palm_x) * screen_width
        raw_y = palm_y * screen_height
        raw_x = max(0, min(raw_x, screen_width - 1))
        raw_y = max(0, min(raw_y, screen_height - 1))

        if self.prev_x is None or self.prev_y is None:
            target_x, target_y = raw_x, raw_y
        else:
            target_x = self.prev_x + (raw_x - self.prev_x) * self.smoothing
            target_y = self.prev_y + (raw_y - self.prev_y) * self.smoothing
        target_x = max(0, min(target_x, screen_width - 1))
        target_y = max(0, min(target_y, screen_height - 1))
        self._log(f"Target: ({target_x:.0f}, {target_y:.0f})")

        if (
            gesture in _CONTINUOUS_GESTURES
            or gesture != self.last_gesture
            or (current_time - self.last_action_time) > self.cooldown
        ):
            if gesture == "move":
                pyautogui.moveTo(int(target_x), int(target_y))
                self._release_all()
                self._log("Mouse moved!")
            elif gesture == "click":
                pyautogui.moveTo(int(target_x), int(target_y))
                pyautogui.click()
                self._log("Left Clicked!")
            elif gesture == "right_click":
                pyautogui.moveTo(int(target_x), int(target_y))
                pyautogui.rightClick()
                self._log("Right Clicked!")
            elif gesture == "double_click":
                pyautogui.moveTo(int(target_x), int(target_y))
                pyautogui.doubleClick()
                self._log("Double Clicked!")
            elif gesture == "drag":
                pyautogui.moveTo(int(target_x), int(target_y))
                if not self.dragging:
                    pyautogui.mouseDown()
                    self.dragging = True
                    self._log("Drag started!")
                else:
                    self._log("Dragging continued!")
            elif gesture == "select":
                pyautogui.moveTo(int(target_x), int(target_y))
                if not self.selecting:
                    pyautogui.mouseDown()
                    self.selecting = True
                    self._log("Selection started!")
                else:
                    self._log("Selecting continued!")
            elif gesture == "hold":
                pyautogui.moveTo(int(target_x), int(target_y))
                if not self.holding:
                    pyautogui.mouseDown()
                    self.holding = True
                    self._log("Hold started!")
                else:
                    self._log("Holding continued!")
            elif gesture == "scroll":
                delta_y = gesture_data["delta_y"] * SCROLL_SPEED
                pyautogui.scroll(int(-delta_y * 100))
                self._log(f"Scrolled! Delta: {delta_y:.2f}")
            elif gesture == "zoom":
                dist = gesture_data["dist"]
                if self.last_dist is not None:
                    zoom_delta = (dist - self.last_dist) * ZOOM_SPEED
                    if zoom_delta > 0:
                        pyautogui.hotkey("ctrl", "+")
                        self._log("Zoomed in!")
                    elif zoom_delta < 0:
                        pyautogui.hotkey("ctrl", "-")
                        self._log("Zoomed out!")
                self.last_dist = dist
            elif gesture == "back":
                pyautogui.hotkey("alt", "left")
                self._log("Went back!")
            elif gesture == "forward":
                pyautogui.hotkey("alt", "right")
                self._log("Went forward!")
            elif gesture in _NOOP_GESTURES:
                pass
            else:
                self._log(f"Ignoring unknown gesture: {gesture}")

            self.last_gesture = gesture
            self.last_action_time = current_time

        self.prev_x, self.prev_y = target_x, target_y

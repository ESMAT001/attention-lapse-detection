"""Blink counts adapted from alireza787b/Python-Gaze-Face-Tracker."""


class BlinkCounter:
    """Counts a blink each time the eyes open again after being closed."""

    def __init__(self, ear_thresh: float) -> None:
        self.ear_thresh = ear_thresh
        self.reset()

    def reset(self) -> None:
        self.closed_streak = 0
        self.total_blinks = 0

    def update(self, ear_score: float | None) -> int:
        blink_now = 0
        # If the face disappears, we keep the closed-eye count where it was.
        if ear_score is not None:
            if ear_score <= self.ear_thresh:
                self.closed_streak += 1
            else:
                # A blink is counted when the eyes reopen, even after one closed frame.
                if self.closed_streak >= 1:
                    self.total_blinks += 1
                    blink_now = 1
                self.closed_streak = 0
        return blink_now

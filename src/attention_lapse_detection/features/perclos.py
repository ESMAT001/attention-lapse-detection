from collections import deque
from attention_lapse_detection.utils.constants import PERCLOS_WINDOW_SECONDS


class PERCLOS:
    """Rolling eye-closure fraction, adapted from e_candeloro_Driver_State_Detection."""

    def __init__(
        self, ear_thresh: float, time_period_s: float = PERCLOS_WINDOW_SECONDS
    ) -> None:
        self.ear_thresh = ear_thresh
        self.time_period_s = time_period_s
        self.reset()

    def reset(self) -> None:
        self.window = deque()

    def update(self, t_now: float, ear_score: float | None) -> float:
        if ear_score is None:
            eye_closed = False
        else:
            eye_closed = ear_score <= self.ear_thresh
        self.window.append((t_now, eye_closed))

        oldest_ok = t_now - self.time_period_s
        while self.window[0][0] < oldest_ok:
            self.window.popleft()

        return sum(closed for _, closed in self.window) / len(self.window)

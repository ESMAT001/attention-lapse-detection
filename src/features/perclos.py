from collections import deque
from attention_lapse_detection.utils.constants import PERCLOS_WINDOW_SECONDS


class PERCLOS:
    """Rolling PERcentage of eye CLOSure over a fixed time period.

    Adapted from e_candeloro_Driver_State_Detection AttentionScorer.get_rolling_PERCLOS.
    """

    def __init__(self, ear_threshold: float, time_period: float = PERCLOS_WINDOW_SECONDS):
        self.ear_threshold = ear_threshold
        self.time_period = time_period
        self.reset()

    def reset(self):
        self.window = deque()

    def update(self,t_now:float, ear: float | None) -> float:
        "Returns the rolling PERCLOS value after updating the window."

        eye_closed = ear is not None and ear <= self.ear_threshold
        self.window.append((t_now,eye_closed))

        while self.window and self.window[0][0] < t_now - self.time_period:
            self.window.popleft()

        return sum(closed for _, closed in self.window) / len(self.window) if self.window else 0.0
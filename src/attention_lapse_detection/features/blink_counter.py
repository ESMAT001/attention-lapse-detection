class BlinkCounter:
    """ Counts blinks from per frame EAR.
        src: dtqAdapted from alireza787b/Python-Gaze-Face-Tracker.
    """

    def __init__(self, ear_threshold: float):
        self.ear_threshold = ear_threshold
        self.reset()

    def reset(self):
        self.eyes_blink_frame_counter = 0
        self.total_blinks = 0

    def update(self, ear: float | None) -> int:
        """
        Return 1 on the frame a blink finishes, else 0.

        `total` is the running count.
        """

        blink_event = 0

        if ear is not None:
            if ear <= self.ear_threshold:
                self.eyes_blink_frame_counter += 1
            else:
                if self.eyes_blink_frame_counter > 0:
                    blink_event = 1
                    self.total_blinks += 1
                self.eyes_blink_frame_counter = 0

        return blink_event

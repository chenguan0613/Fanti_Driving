from collections import deque
from typing import List
from src.features import FrameFeature


class SlidingWindowBuffer:
    """
    Maintain a fixed-length time sliding window to store
    the FrameFeatures of the most recent N frames.
    When the set window size is reached, the complete sequence
    can be output for the Aggregator to calculate the temporal features.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.buffer: deque[FrameFeature] = deque(maxlen=window_size)

    def push(self, feature: FrameFeature) -> None:
        # Push new frames
        self.buffer.append(feature)

    def is_ready(self) -> bool:
        # Check whether the window is full or not
        return len(self.buffer) == self.window_size

    def get_window(self) -> List[FrameFeature]:
        # Obtain the data in the entire window
        return list(self.buffer)

    def clear(self) -> None:
        # Reset the buffer
        self.buffer.clear()

    @property
    def current_size(self) -> int:
        # Obtain how many frames in the buffer currently
        return len(self.buffer)

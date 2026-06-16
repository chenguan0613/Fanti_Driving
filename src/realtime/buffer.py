from collections import deque
from typing import List
from src.features.frame_schema import FrameFeature
class SlidingWindowBuffer:
    """
    维护一个固定长度的时间滑窗，用于存储最近 N 帧的 FrameFeature。
    当达到设定的窗口大小时，即可输出完整序列供 Aggregator 计算时序特征。
    """
    def __init__(self, window_size: int = 30):
        """
        初始化缓冲区
        :param window_size: 窗口容量。若视频为 10 fps，30 帧即代表过去 3 秒的记忆。
        """
        self.window_size = window_size
        self.buffer: deque[FrameFeature] = deque(maxlen=window_size)
    def push(self, feature: FrameFeature)->None:
        self.buffer.append(feature)
    
    def is_ready(self) -> bool:
        return len(self.buffer) == self.window_size
    
    def get_window(self) -> List[FrameFeature]:
        return list(self.buffer)
    
    def clear(self) -> None:
        self.buffer.clear()
    
    @property
    def current_size(self) -> int:
        return len(self.buffer)


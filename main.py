from src.preprocessing.frame_extractor import FrameExtractor
from src.preprocessing.video_loader import VideoLoader

loader = VideoLoader("./data/raw_videos/1.mp4", target_fps=10)
extractor = FrameExtractor("./face_landmarker.task")

for i, frame in loader.frame():
    features = extractor.extract(frame, timestamp=int(i / 10))
    print(features)

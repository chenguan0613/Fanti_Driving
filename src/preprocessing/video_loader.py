import cv2


class VideoLoader:
    """
    Read a video under `video_path`, and divide it into frames.
    """

    def __init__(self, video_path: str, target_fps: int = 10) -> None:
        self.video_path = video_path
        self.target_fps = target_fps

        self.cap = cv2.VideoCapture(video_path)
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = int(self.source_fps / target_fps)

        if self.frame_interval <= 0:
            self.frame_interval = 1

    def frame(self):
        frame_id = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_id % self.frame_interval == 0:
                yield frame_id, frame

            frame_id += 1

        self.cap.release()

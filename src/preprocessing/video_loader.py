import cv2


class VideoLoader:
    """
    Read a video under `video_path`, divide it into frames, do frame dropping(if needed)
    and accurately yield frame_id, timestamp (seconds), and the frame itself.
    """

    def __init__(self, video_path: str, target_fps: int = 10) -> None:
        # Define the path and fps of the video loaded. (Default: 10 frame per second)
        self.video_path = video_path
        self.target_fps = target_fps
        # Use OpenCV to instantiate a video capture object
        self.cap = cv2.VideoCapture(video_path)
        # Obtain the original fps
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Convert the fps into our target fps
        self.frame_interval = int(self.source_fps / target_fps)
        # Avoid the case of source_fps<target_fps
        self.frame_interval = max(1, int(self.source_fps / target_fps))

    def frame(self):
        # Initialize the frame counter
        frame_id = 0

        while True:
            # ret: check whether read the frame or not
            # frame: detail data of single frame
            ret, frame = self.cap.read()
            # If cannot obtain frame anymore, it means that the video ends
            if not ret:
                break
            # Actual frame dropping operation
            if frame_id % self.frame_interval == 0:
                timestamp_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield frame_id, timestamp_sec, frame
            # Goto next frame
            frame_id += 1

        self.cap.release()

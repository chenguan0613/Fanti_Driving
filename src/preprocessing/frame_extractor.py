import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from src.features.frame_schema import FrameFeature

# TODO:
# - Detect head actions and gaze information.


class FrameExtractor:
    """
    Extract frame features for further train.
    """

    def __init__(self, model_path="face_landmarker.task") -> None:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract(self, frame, timestamp=0) -> FrameFeature:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.detector.detect(mp_image)

        if not self._has_face(result):
            return FrameFeature(timestamp=timestamp, face_detected=0)

        lm: list = self._get_landmarks(result)
        ear, left_ear, right_ear = self._compute_ear(lm)
        mar = self._compute_mar(lm)
        return self._build_feature(
            timestamp=timestamp,
            ear=ear,
            left_ear=left_ear,
            right_ear=right_ear,
            mar=mar,
        )

    def _to_mp_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _detect(self, mp_image):
        return self.detector.detect(mp_image)

    def _has_face(self, result):
        return bool(result.face_landmarks)

    def _get_landmarks(self, result) -> list:
        return result.face_landmarks[0]

    def _get_point(self, lm: list, i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    def _compute_ear(self, lm: list) -> tuple[float, float, float]:
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        def ear(eye: list[int]) -> float:
            v1 = np.linalg.norm(
                self._get_point(lm, eye[1]) - self._get_point(lm, eye[5])
            )
            v2 = np.linalg.norm(
                self._get_point(lm, eye[2]) - self._get_point(lm, eye[4])
            )
            h = np.linalg.norm(
                self._get_point(lm, eye[0]) - self._get_point(lm, eye[3])
            )
            return float((v1 + v2) / (2.0 * h + 1e-6))

        left_ear = ear(LEFT_EYE)
        righ_ear = ear(RIGHT_EYE)
        avg_ear = (left_ear + righ_ear) / 2
        return avg_ear, left_ear, righ_ear

    def _compute_mar(self, lm: list) -> float:
        MOUTH = [61, 291, 13, 14]

        p = lambda i: self._get_point(lm, i)
        vertical = np.linalg.norm(p(MOUTH[2]) - p(MOUTH[3]))
        horizontal = np.linalg.norm(p(MOUTH[0]) - p(MOUTH[1]))
        return float(vertical / (horizontal + 1e-6))

    def _build_feature(
        self,
        timestamp: float,
        ear: float,
        left_ear: float,
        right_ear: float,
        mar: float,
    ) -> FrameFeature:
        eye_closed = 1 if ear < 0.2 else 0
        return FrameFeature(
            timestamp=timestamp,
            face_detected=1,
            left_EAR=left_ear,
            right_EAR=right_ear,
            mounth_open_ratio=mar,
            head_pitch=0,
            head_yaw=0,
            gaze_x=0,
            gaze_y=0,
            eye_closed=eye_closed,
        )

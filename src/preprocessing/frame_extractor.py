import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from src.features import FrameFeature


class FrameExtractor:
    """
    Extracting human features using Mediapipe:
    - Head features: pitch, yaw
    - Facial features: MAR (mouth), EAR (eye), gaze
    """

    def __init__(self, model_path="face_landmarker.task") -> None:
        # Load an existing face recognition model and obtain relevant parameters
        base_options = python.BaseOptions(model_asset_path=model_path)
        # Set parameters
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        # Send the frame to the detector
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def close(self):
        """
        `close()` should be called after `extrac()`.
        """
        if self.detector:
            self.detector.close()

    def extract(self, frame, timestamp=0.0) -> FrameFeature:
        # Convert the image into Mediapipe object
        mp_image = self._to_mp_image(frame)
        result = self._detect(mp_image)
        # Obtain the key points of frame
        if not self._has_face(result):
            return FrameFeature(timestamp=timestamp, face_detected=0)
        # extract landmark
        lm: list = self._get_landmarks(result)
        # calculate eye opening and closing
        ear, left_ear, right_ear = self._compute_ear(lm)
        # calculate mouth open ratio
        mar = self._compute_mar(lm)
        # extract the head pose: Pitch and Yaw
        pitch, yaw, _ = self._compute_head_pose(result)
        # Estimating the focal point of the line of sight
        gaze_x, gaze_y = self._compute_gaze(lm)

        return self._build_feature(
            timestamp=timestamp,
            ear=ear,
            left_ear=left_ear,
            right_ear=right_ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            landmarks=lm,
        )

    def _has_face(self, result):
        return bool(result.face_landmarks)

    def _get_landmarks(self, result) -> list:
        return result.face_landmarks[0]

    def _get_point(self, lm: list, i: int) -> np.ndarray:
        return np.array([lm[i].x, lm[i].y], dtype=np.float32)

    def _compute_ear(self, lm: list) -> tuple[float, float, float]:
        # 0: inner eye, 3: outer eye
        # 1,2: upper eyelid
        # 4,5: lower eyelid
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        def ear(eye: list[int]) -> float:
            # Distance between the upper eyelid 1 and lower eyelid 5
            v1 = np.linalg.norm(
                self._get_point(lm, eye[1]) - self._get_point(lm, eye[5])
            )
            # Distance between the upper eyelid 2 and lower eyelid 4
            v2 = np.linalg.norm(
                self._get_point(lm, eye[2]) - self._get_point(lm, eye[4])
            )
            # Distance between inner corner of eye 0 and outer corner of the eye 3
            h = np.linalg.norm(
                self._get_point(lm, eye[0]) - self._get_point(lm, eye[3])
            )
            # EAR formular
            return float((v1 + v2) / (2.0 * h + 1e-6))

        # EAR of left eye and right eye
        left_ear = ear(LEFT_EYE)
        righ_ear = ear(RIGHT_EYE)
        # Average eye opening and closing
        avg_ear = (left_ear + righ_ear) / 2
        return avg_ear, left_ear, righ_ear

    def _compute_mar(self, lm: list) -> float:
        # 0: left corner of mouth, 1: right corner of mouth
        # 3: center of the inner edge of the upper lip
        # 4: center of the inner edge of the lower lip
        MOUTH = [61, 291, 13, 14]
        p = lambda i: self._get_point(lm, i)
        # Vertical distance of upper and lower lip
        vertical = np.linalg.norm(p(MOUTH[2]) - p(MOUTH[3]))
        # Horizontal distance of left and right corner of mouth
        horizontal = np.linalg.norm(p(MOUTH[0]) - p(MOUTH[1]))
        # MAR formular
        return float(vertical / (horizontal + 1e-6))

    def _compute_head_pose(self, result) -> tuple[float, float, float]:
        # obtain pitch, yaw and roll using OpenCV
        matrix = result.facial_transformation_matrixes[0]
        rotation_matrix = matrix[:3, :3]
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, roll = angles
        return float(pitch), float(yaw), float(roll)

    def _compute_gaze(self, lm: list) -> tuple[float, float]:
        LEFT_IRIS = [468, 469, 470, 471, 472]
        RIGHT_IRIS = [473, 474, 475, 476, 477]

        LEFT_EYE_CENTER = (33, 133)
        RIGHT_EYE_CENTER = (362, 263)

        def center(i1, i2):
            return (self._get_point(lm, i1) + self._get_point(lm, i2)) / 2.0

        left_center = center(*LEFT_EYE_CENTER)
        right_center = center(*RIGHT_EYE_CENTER)

        left_iris = np.mean([self._get_point(lm, i) for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([self._get_point(lm, i) for i in RIGHT_IRIS], axis=0)

        gaze = (left_iris + right_iris) / 2.0
        eye_center = (left_center + right_center) / 2.0

        gaze_vec = gaze - eye_center

        return float(gaze_vec[0]), float(gaze_vec[1])

    def _to_mp_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _detect(self, mp_image):
        return self.detector.detect(mp_image)

    def _build_feature(
        self,
        timestamp: float,
        ear: float,
        left_ear: float,
        right_ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        gaze_x: float,
        gaze_y: float,
        landmarks: list,
    ) -> FrameFeature:
        ear = min(ear, 0.6)
        left_ear = min(left_ear, 0.6)
        right_ear = min(right_ear, 0.6)
        eye_closed = 1 if ear < 0.18 else 0
        return FrameFeature(
            timestamp=timestamp,
            face_detected=1,
            left_EAR=left_ear,
            right_EAR=right_ear,
            mouth_open_ratio=mar,
            head_pitch=pitch,
            head_yaw=yaw,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            eye_closed=eye_closed,
            raw_landmarks=landmarks,
        )

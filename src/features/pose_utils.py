import cv2
import numpy as np
class PoseUtils:
    @staticmethod
    def get_euler_angles(transformation_matrix) -> tuple[float, float, float]:
        rotation_matrix = transformation_matrix[:3, :3]
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        pitch = angles[0] * 360 
        yaw = angles[1] * 360
        roll = angles[2] * 360
        return pitch, yaw, roll
    
    @staticmethod
    def get_gaze_center(landmarks) -> tuple[float, float]:
        try:
            left_iris = np.array([landmarks[473].x, landmarks[473].y], dtype=np.float32)
            right_iris = np.array([landmarks[468].x, landmarks[468].y], dtype=np.float32)
            gaze_x = float((left_iris[0] + right_iris[0]) / 2.0)
            gaze_y = float((left_iris[1] + right_iris[1]) / 2.0)
            return gaze_x, gaze_y
        except IndexError:
            return 0.0, 0.0
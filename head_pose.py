import cv2
import numpy as np

import config


class HeadPoseEstimator:
    def __init__(self):
        self.camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    def estimate(self, landmarks, frame_shape):
        if landmarks is None or len(landmarks) != 68:
            return {
                "state": "UNCERTAIN",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "confidence": 0.0,
            }
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if not np.isfinite(landmarks).all():
            return {
                "state": "UNCERTAIN",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "confidence": 0.0,
            }

        image_points = np.array([
            landmarks[17],
            landmarks[21],
            landmarks[22],
            landmarks[26],
            landmarks[36],
            landmarks[45],
            landmarks[48],
            landmarks[54],
        ], dtype=np.float32)

        model_3d = np.array([
            [0.0, 0.0, 0.0],
            [-30.0, 0.0, -10.0],
            [30.0, 0.0, -10.0],
            [0.0, 35.0, -15.0],
            [-50.0, 50.0, -20.0],
            [50.0, 50.0, -20.0],
            [0.0, 80.0, -20.0],
            [0.0, 130.0, -20.0],
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            model_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {
                "state": "UNCERTAIN",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "confidence": 0.0,
            }

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        if sy > 1e-6:
            yaw = np.rad2deg(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
            pitch = np.rad2deg(np.arctan2(-rotation_matrix[2, 0], sy))
            roll = np.rad2deg(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        else:
            yaw = np.rad2deg(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
            pitch = np.rad2deg(np.arctan2(-rotation_matrix[2, 0], sy))
            roll = np.rad2deg(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))

        if abs(pitch) > config.HEAD_DOWN_THRESHOLD:
            state = "DOWN"
        elif yaw > config.HEAD_DOWN_THRESHOLD:
            state = "RIGHT"
        elif yaw < -config.HEAD_DOWN_THRESHOLD:
            state = "LEFT"
        else:
            state = "NORMAL"

        return {
            "state": state,
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "confidence": 1.0,
        }

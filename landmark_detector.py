import os

import dlib
import numpy as np


class LandmarkDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"dlib landmark model not found: {self.model_path}")
        self.predictor = dlib.shape_predictor(self.model_path)

    def detect(self, frame, face_rect):
        if face_rect is None:
            return None

        x = int(face_rect["x"])
        y = int(face_rect["y"])
        width = int(face_rect["width"])
        height = int(face_rect["height"])

        if width <= 0 or height <= 0:
            return None

        rect = dlib.rectangle(x, y, x + width, y + height)
        shape = self.predictor(frame, rect)
        points = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        return points

import os

import cv2

from config import (
    FACE_CONFIDENCE_THRESHOLD,
    FACE_NMS_THRESHOLD,
    FACE_TOP_K,
    MODEL_YUNET,
)


class YuNetFaceDetector:
    def __init__(self, model_path=None, confidence_threshold=None, nms_threshold=None):
        self.model_path = model_path or MODEL_YUNET
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YuNet model not found: {self.model_path}")

        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else FACE_CONFIDENCE_THRESHOLD
        self.nms_threshold = nms_threshold if nms_threshold is not None else FACE_NMS_THRESHOLD
        self.detector = cv2.FaceDetectorYN_create(
            self.model_path,
            "",
            (320, 320),
            self.confidence_threshold,
            self.nms_threshold,
            FACE_TOP_K,
        )
        self.loaded = True

    def detect_faces(self, frame):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("YuNet expects a non-empty three-channel BGR frame.")
        h, w = frame.shape[:2]
        input_size = (w, h)
        if self.detector.getInputSize() != input_size:
            self.detector.setInputSize(input_size)
        detection = self.detector.detect(frame)
        if not isinstance(detection, tuple) or len(detection) < 2:
            raise RuntimeError("YuNet returned an unexpected detection result.")

        _, faces = detection
        if faces is None:
            return []

        results = []
        for face in faces:
            if face is None or len(face) < 15:
                continue
            x, y, width, height = [float(v) for v in face[:4]]
            confidence = float(face[-1])
            if confidence < self.confidence_threshold:
                continue

            x1 = max(0, min(w - 1, int(round(x))))
            y1 = max(0, min(h - 1, int(round(y))))
            x2 = max(x1 + 1, min(w, int(round(x + width))))
            y2 = max(y1 + 1, min(h, int(round(y + height))))
            results.append({
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": confidence,
            })

        results.sort(key=lambda item: (item["width"] * item["height"], item["confidence"]), reverse=True)
        return results

    def draw_faces(self, frame, faces, selected_index=None):
        for idx, face in enumerate(faces):
            x = int(face["x"])
            y = int(face["y"])
            width = int(face["width"])
            height = int(face["height"])
            color = (0, 255, 0) if selected_index == idx else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)

            label = "DRIVER" if selected_index == idx else "OTHER FACE"
            text_pos = (x, max(20, y - 10))
            cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{label} {face['confidence']:.3f}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return frame

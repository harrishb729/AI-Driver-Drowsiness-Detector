import logging
import os
import time
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np

import config
from alert_system import AlertSystem
from dashboard import Dashboard
from drowsiness_engine import DrowsinessEngine
from eye_analyzer import EyeAnalyzer
from face_detector import YuNetFaceDetector
from head_pose import HeadPoseEstimator
from landmark_detector import LandmarkDetector
from mouth_analyzer import MouthAnalyzer
from session_logger import SessionLogger
from utils import format_elapsed
from version import APP_NAME, APP_VERSION


LOGGER = logging.getLogger("driver_drowsiness")


def configure_application_logging():
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in LOGGER.handlers[:]:
        LOGGER.removeHandler(handler)
        handler.close()

    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        handler = RotatingFileHandler(
            config.APPLICATION_LOG_PATH,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"WARNING: Application file logging is unavailable: {exc}")
        handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    LOGGER.addHandler(handler)
    return LOGGER


class DriverDrowsinessApp:
    def __init__(self):
        self.detector = None
        self.landmark_detector = None
        self.eye_analyzer = EyeAnalyzer()
        self.mouth_analyzer = MouthAnalyzer()
        self.head_pose = HeadPoseEstimator()
        self.engine = DrowsinessEngine()
        self.alert = AlertSystem()
        self.logger = SessionLogger(config.SESSION_LOG_DIR)
        self.dashboard = Dashboard()
        self.camera = None
        self.detected_faces = []
        self.selected_face = None
        self.last_detection_time = 0.0
        self.driver_label = "FACE NOT DETECTED"
        self.previous_driver_face = None
        self.last_status = "NORMAL"
        self.warning_logged = False
        self.last_head_down_logged = False
        self.last_sustained_eye_logged = False
        self.head_down_started = None
        self.face_missing_started = None
        self.face_loss_reset_done = False
        self.last_perception = None
        self.last_eye_result = None
        self.last_mouth_result = None
        self.last_head_result = None
        self.start_time = time.monotonic()
        self.calibration_started = None
        self.calibration_active = False
        self.has_calibrated = False
        self.calibration_failed = False
        self.calibration_state = "WAITING"
        self.pending_recalibration = False
        self._shutdown_complete = False

    def initialize(self):
        print(f"{APP_NAME.upper()} | VERSION {APP_VERSION}")
        print("Loading AI models...")
        LOGGER.info("Startup requested")

        missing = []
        for label, path in {"YuNet": config.MODEL_YUNET, "dlib": config.MODEL_DLIB}.items():
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")

        if missing:
            print("Required model files are missing:")
            for item in missing:
                print(" -", item)
            raise FileNotFoundError("Missing required model files.")

        self.detector = YuNetFaceDetector(
            config.MODEL_YUNET,
            config.FACE_CONFIDENCE_THRESHOLD,
            config.FACE_NMS_THRESHOLD,
        )
        self.landmark_detector = LandmarkDetector(config.MODEL_DLIB)
        LOGGER.info("YuNet and dlib landmark models loaded")

        print("Initializing camera...")
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.camera.isOpened():
            raise RuntimeError("Unable to open the default webcam. Check camera connectivity and permissions.")

        success, frame = self.camera.read()
        if not success or frame is None:
            raise RuntimeError("Failed to read frames from the webcam.")

        print(f"CAMERA: OK | FRAME: {frame.shape[1]} x {frame.shape[0]}")
        LOGGER.info("Camera initialized at %s x %s", frame.shape[1], frame.shape[0])
        print(
            f"YUNET: LOADED | confidence={self.detector.confidence_threshold:.2f} "
            f"| NMS={self.detector.nms_threshold:.2f}"
        )
        print("SYSTEM READY")
        LOGGER.info("System ready; calibration will begin when a valid face is detected")

    def select_probable_driver(self, faces):
        if not faces:
            self.selected_face = None
            self.previous_driver_face = None
            self.driver_label = "FACE NOT DETECTED"
            return None

        frame_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        center_x = frame_w / 2.0
        center_y = frame_h / 2.0

        def intersection_over_union(first, second):
            left = max(first["x"], second["x"])
            top = max(first["y"], second["y"])
            right = min(first["x"] + first["width"], second["x"] + second["width"])
            bottom = min(first["y"] + first["height"], second["y"] + second["height"])
            intersection = max(0, right - left) * max(0, bottom - top)
            first_area = first["width"] * first["height"]
            second_area = second["width"] * second["height"]
            union = first_area + second_area - intersection
            return intersection / union if union else 0.0

        best_face = None
        best_score = float("-inf")
        for face in faces:
            face_center_x = face["x"] + face["width"] / 2.0
            face_center_y = face["y"] + face["height"] / 2.0
            distance = abs(face_center_x - center_x) + abs(face_center_y - center_y) * 0.6
            area = face["width"] * face["height"]
            score = area - (distance * 0.35)
            if self.previous_driver_face is not None:
                overlap = intersection_over_union(face, self.previous_driver_face)
                score += overlap * area * 0.75
            if score > best_score:
                best_score = score
                best_face = face

        self.selected_face = best_face
        self.previous_driver_face = best_face.copy()
        return best_face

    @staticmethod
    def landmarks_are_valid(landmarks, face, frame_shape):
        if landmarks is None or face is None:
            return False
        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (68, 2) or not np.isfinite(points).all():
            return False
        if (
            face["width"] < config.MIN_FACE_WIDTH
            or face["height"] < config.MIN_FACE_HEIGHT
        ):
            return False

        frame_height, frame_width = frame_shape[:2]
        margin_x = face["width"] * config.LANDMARK_FACE_MARGIN_RATIO
        margin_y = face["height"] * config.LANDMARK_FACE_MARGIN_RATIO
        left = max(0.0, face["x"] - margin_x)
        top = max(0.0, face["y"] - margin_y)
        right = min(float(frame_width), face["x"] + face["width"] + margin_x)
        bottom = min(float(frame_height), face["y"] + face["height"] + margin_y)
        if (
            np.any(points[:, 0] < left)
            or np.any(points[:, 0] > right)
            or np.any(points[:, 1] < top)
            or np.any(points[:, 1] > bottom)
        ):
            return False

        left_eye_width = np.linalg.norm(points[36] - points[39])
        right_eye_width = np.linalg.norm(points[42] - points[45])
        mouth_width = np.linalg.norm(points[48] - points[54])
        return (
            left_eye_width >= config.MIN_EYE_WIDTH
            and right_eye_width >= config.MIN_EYE_WIDTH
            and mouth_width >= config.MIN_MOUTH_WIDTH
        )

    def calibration_loop(self, frame, landmarks):
        if self.pending_recalibration:
            self.eye_analyzer.reset_calibration()
            self.calibration_started = None
            self.calibration_active = False
            self.has_calibrated = False
            self.calibration_failed = False
            self.pending_recalibration = False

        if landmarks is None or len(landmarks) != 68:
            self.calibration_active = False
            self.calibration_state = "WAITING FOR FACE"
            return

        if self.calibration_started is None:
            self.calibration_started = time.monotonic()
            print("Calibrating...")
            LOGGER.info("Calibration started")
        self.calibration_active = True
        self.calibration_state = "CALIBRATING"
        self.eye_analyzer.add_calibration_sample(landmarks)
        self.mouth_analyzer.add_calibration_sample(landmarks)

        elapsed = time.monotonic() - self.calibration_started
        if elapsed >= config.CALIBRATION_TIME:
            if self.eye_analyzer.finalize_calibration():
                self.calibration_active = False
                self.calibration_failed = False
                self.has_calibrated = True
                self.mouth_analyzer.finalize_calibration()
                self.calibration_state = "READY"
                print("CALIBRATION COMPLETE")
                LOGGER.info("Calibration completed")
            else:
                self.calibration_failed = True
                self.calibration_state = "FAILED"
                print("CALIBRATION FAILED — LOOK AT CAMERA")
                self.calibration_started = time.monotonic()

    def run(self):
        try:
            self.initialize()
        except Exception as exc:
            print(f"ERROR: {exc}")
            LOGGER.exception("Application initialization failed")
            return 1

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, self.dashboard.WIDTH, self.dashboard.HEIGHT)
        LOGGER.info("Monitoring started")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("ERROR: Camera frame acquisition failed. Shutting down.")
                LOGGER.error("Camera frame acquisition failed")
                return 1

            frame = cv2.flip(frame, 1)
            now = time.monotonic()

            if now - self.last_detection_time >= config.FACE_DETECTION_INTERVAL:
                self.detected_faces = self.detector.detect_faces(frame)
                self.last_detection_time = now

            selected_face = self.select_probable_driver(self.detected_faces)
            selected_index = None
            if selected_face is not None and self.detected_faces:
                selected_index = self.detected_faces.index(selected_face)
            self.detector.draw_faces(frame, self.detected_faces, selected_index)

            landmarks = None
            if selected_face is not None:
                landmarks = self.landmark_detector.detect(frame, selected_face)
                self.driver_label = "PROBABLE DRIVER"
            else:
                self.driver_label = "FACE NOT DETECTED"

            landmarks_valid = self.landmarks_are_valid(landmarks, selected_face, frame.shape)
            face_valid = selected_face is not None
            perception_valid = face_valid and landmarks_valid
            eye_state = "EYES UNCERTAIN"
            mouth_state = "MOUTH UNCERTAIN"
            head_state = "UNCERTAIN"
            eye_result = {
                "state": eye_state,
                "left_ear": None,
                "right_ear": None,
                "blink_detected": False,
                "closure_duration": 0.0,
                "sustained_closure_duration": 0.0,
                "tracking_valid": False,
            }
            mouth_result = {
                "state": mouth_state,
                "mouth_ratio": None,
                "yawn_duration": 0.0,
                "yawn_detected": False,
                "tracking_valid": False,
            }
            head_result = {
                "state": head_state,
                "pitch": None,
                "yaw": None,
                "confidence": 0.0,
            }
            score = self.engine.score
            driver_status = self.engine.status

            if perception_valid:
                self.face_missing_started = None
                self.face_loss_reset_done = False
            else:
                if self.face_missing_started is None:
                    self.face_missing_started = now
                if (
                    now - self.face_missing_started >= config.FACE_LOSS_RESET_TIME
                    and not self.face_loss_reset_done
                ):
                    self.eye_analyzer.reset_tracking()
                    self.mouth_analyzer.reset_tracking()
                    self.head_down_started = None
                    self.face_loss_reset_done = True

            calibration_frame = not self.has_calibrated
            if calibration_frame:
                self.calibration_loop(
                    frame,
                    landmarks if perception_valid else None,
                )
            else:
                if perception_valid:
                    eye_result = self.eye_analyzer.analyze(landmarks, now)
                    mouth_result = self.mouth_analyzer.update(landmarks, now)
                    head_result = self.head_pose.estimate(landmarks, frame.shape)
                    eye_state = eye_result["state"]
                    mouth_state = mouth_result["state"]
                    head_state = head_result["state"]
                    self.last_perception = {
                        "eye_state": eye_state,
                        "mouth_state": mouth_state,
                        "head_state": head_state,
                    }
                    self.last_eye_result = eye_result
                    self.last_mouth_result = mouth_result
                    self.last_head_result = head_result
                elif (
                    self.face_missing_started is not None
                    and now - self.face_missing_started <= config.FACE_LOST_GRACE_TIME
                    and self.last_perception is not None
                ):
                    eye_state = self.last_perception["eye_state"]
                    mouth_state = self.last_perception["mouth_state"]
                    head_state = self.last_perception["head_state"]

                if perception_valid and head_state == "DOWN":
                    if self.head_down_started is None:
                        self.head_down_started = now
                    head_down_duration = now - self.head_down_started
                else:
                    if perception_valid or self.face_loss_reset_done:
                        self.head_down_started = None
                    head_down_duration = 0.0

                if (
                    perception_valid
                    and eye_state == "EYES CLOSED"
                    and eye_result["closure_duration"] >= config.EYE_CLOSURE_CONFIRM_TIME
                ):
                    if not self.last_sustained_eye_logged:
                        self.last_sustained_eye_logged = True
                        self.logger.log_event(
                            "SUSTAINED_EYE_CLOSURE",
                            float(self.engine.score),
                            eye_state,
                            mouth_state,
                            head_state,
                        )
                elif perception_valid:
                    self.last_sustained_eye_logged = False

                if (
                    perception_valid
                    and head_down_duration >= config.HEAD_DOWN_CONFIRM_TIME
                    and not self.last_head_down_logged
                ):
                    self.logger.log_event(
                        "HEAD_DOWN",
                        float(self.engine.score),
                        eye_state,
                        mouth_state,
                        head_state,
                    )
                    self.last_head_down_logged = True
                elif perception_valid and head_state != "DOWN":
                    self.last_head_down_logged = False

                engine_output = self.engine.update(
                    eye_state=eye_state if perception_valid else "EYES UNCERTAIN",
                    eye_closure_duration=eye_result["closure_duration"],
                    blink_event=eye_result["blink_detected"] if perception_valid else False,
                    head_state=head_state if perception_valid else "UNCERTAIN",
                    head_down_duration=head_down_duration,
                    yawn_detected=mouth_result["yawn_detected"] if perception_valid else False,
                    yawn_duration=mouth_result["yawn_duration"],
                    landmark_valid=landmarks_valid,
                    face_valid=face_valid,
                    timestamp=now,
                )
                score = engine_output["score"]
                driver_status = engine_output["status"]

                if perception_valid and eye_result["blink_detected"]:
                    self.logger.log_event("BLINK", float(score), eye_state, mouth_state, head_state)
                if perception_valid and mouth_result["yawn_detected"]:
                    self.logger.log_event("YAWN", float(score), eye_state, mouth_state, head_state)

                for event in engine_output["events"]:
                    if event in ("BLINK", "YAWN"):
                        continue
                    self.logger.log_event(
                        event,
                        float(score),
                        eye_state,
                        mouth_state,
                        head_state,
                    )

                if engine_output["alarm"] and not self.alert.is_alarm_active():
                    self.alert.start_alarm()
                    LOGGER.warning("Alarm activated")
                elif not engine_output["alarm"] and self.alert.is_alarm_active():
                    self.alert.stop_alarm()
                    LOGGER.info("Alarm cleared")
                self.alert.update(now)
                self.last_status = driver_status

            if calibration_frame:
                engine_output = self.engine.update(
                    eye_state="EYES UNCERTAIN",
                    face_valid=False,
                    landmark_valid=False,
                    timestamp=now,
                )
                score = engine_output["score"]
                driver_status = engine_output["status"]
                for event in engine_output["events"]:
                    self.logger.log_event(
                        event,
                        float(score),
                        eye_state,
                        mouth_state,
                        head_state,
                    )
                if engine_output["alarm"] and not self.alert.is_alarm_active():
                    self.alert.start_alarm()
                    LOGGER.warning("Alarm activated")
                elif not engine_output["alarm"] and self.alert.is_alarm_active():
                    self.alert.stop_alarm()
                    LOGGER.info("Alarm cleared")
                self.alert.update(now)

            session_time = format_elapsed(time.monotonic() - self.start_time)
            event_count = self.logger.event_count

            dashboard_data = {
                "driver_status": driver_status,
                "drowsiness_score": score,
                "eye_state": eye_state,
                "blink_count": self.eye_analyzer.blink_counter,
                "yawn_count": self.mouth_analyzer.yawn_count,
                "head_state": head_state,
                "alarm_state": "ACTIVE" if self.alert.is_alarm_active() else "OFF",
                "session_elapsed": session_time,
                "event_count": event_count,
                "driver_label": self.driver_label,
                "calibration_state": self.calibration_state,
            }

            debug_lines = []
            if config.DEBUG_MODE:
                best_confidence = max(
                    (face["confidence"] for face in self.detected_faces),
                    default=0.0,
                )
                components = engine_output["score_components"]
                debug_lines = [
                    "CAMERA: OK",
                    f"FRAME: {frame.shape[1]} x {frame.shape[0]}",
                    f"YUNET: {'LOADED' if self.detector and self.detector.loaded else 'FAILED'}",
                    f"FACES DETECTED: {len(self.detected_faces)}",
                    f"BEST FACE CONFIDENCE: {best_confidence:.3f}",
                    f"DRIVER FACE: {'FOUND' if selected_face is not None and landmarks_valid else 'NOT FOUND'}",
                    f"EYE STATE: {eye_state} | L/R: {eye_result['left_ear']} / {eye_result['right_ear']}",
                    f"CLOSURE: {eye_result['closure_duration']:.2f}s | PERCLOS: {engine_output['perclos']:.3f}",
                    f"MOUTH: {mouth_state} | RATIO: {mouth_result['mouth_ratio']} | DURATION: {mouth_result['yawn_duration']:.2f}s",
                    f"HEAD: {head_state} | PITCH: {head_result['pitch']} | YAW: {head_result['yaw']}",
                    f"SCORE PARTS: {components}",
                    f"ENGINE REASON: {engine_output['reason']}",
                ]
            dashboard_data["debug_lines"] = debug_lines

            try:
                _, _, window_width, window_height = cv2.getWindowImageRect(config.WINDOW_NAME)
            except cv2.error:
                window_width, window_height = self.dashboard.WIDTH, self.dashboard.HEIGHT

            frame = self.dashboard.render(
                frame,
                dashboard_data,
                debug_mode=config.DEBUG_MODE,
                output_size=(window_width, window_height),
            )
            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                LOGGER.info("Quit requested from dashboard keyboard")
                break
            elif key == ord("r"):
                self.pending_recalibration = True
                self.calibration_active = False
                self.has_calibrated = False
                self.calibration_failed = False
                self.calibration_state = "WAITING"
                self.eye_analyzer.reset_calibration()
                self.mouth_analyzer.reset_calibration()
                self.calibration_started = None
            elif key == ord("d"):
                config.DEBUG_MODE = not config.DEBUG_MODE

            if self.dashboard_window_closed():
                break

        return 0

    @staticmethod
    def dashboard_window_closed():
        try:
            if cv2.getWindowProperty(
                config.WINDOW_NAME,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                LOGGER.info("Dashboard window closed by user")
                return True
        except cv2.error as exc:
            LOGGER.info("Dashboard window is no longer available: %s", exc)
            return True
        return False

    def shutdown(self):
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        LOGGER.info("Shutdown started")
        if self.alert is not None:
            self.alert.stop_alarm()
        if self.camera is not None:
            try:
                self.camera.release()
            except cv2.error:
                LOGGER.exception("Failed to release the camera during shutdown")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            LOGGER.exception("Failed to close OpenCV windows during shutdown")
        if self.logger is not None:
            self.logger.close()
        LOGGER.info("Shutdown complete")

def main():
    logger = configure_application_logging()
    logger.info("%s %s launching", APP_NAME, APP_VERSION)
    app = None
    exit_code = 0
    try:
        app = DriverDrowsinessApp()
        exit_code = app.run()
    except KeyboardInterrupt:
        print("Shutdown requested (Ctrl+C).")
        logger.info("Quit requested from Ctrl+C")
        exit_code = 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        logger.exception("Unexpected application failure")
        exit_code = 1
    finally:
        if app is not None:
            app.shutdown()
        logger.info("Application exited with code %s", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

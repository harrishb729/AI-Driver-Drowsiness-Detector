import time

import cv2
import numpy as np

import config
from alert_system import AlertSystem
from drowsiness_engine import DrowsinessEngine
from eye_analyzer import EyeAnalyzer
from face_detector import YuNetFaceDetector
from head_pose import HeadPoseEstimator
from landmark_detector import LandmarkDetector
from main import DriverDrowsinessApp
from mouth_analyzer import MouthAnalyzer


WINDOW_NAME = "Manual Driver Behavior Validation"
OUTPUT_SIZE = (1440, 1000)
PANEL_WIDTH = 510
PANEL_BACKGROUND = (23, 29, 38)
PANEL_CARD = (34, 42, 53)
TEXT = (237, 242, 248)
MUTED = (156, 169, 184)
GREEN = (98, 218, 145)
AMBER = (30, 185, 255)
RED = (80, 80, 238)
BLUE = (232, 185, 116)

TEST_INSTRUCTIONS = (
    "1 / A — OPEN EYES, MOUTH NEUTRAL (5s)",
    "2 / B — ONE SLOW DELIBERATE BLINK (4s)",
    "3 / C — CLOSE BOTH EYES FOR 2s (4s)",
    "4 / D — MOUTH NEUTRAL (5s)",
    "5 / E — SMILE (3s)",
    "6 / F — TALK NORMALLY (3s)",
    "7 / G — WIDE YAWN, THEN CLOSE (4s)",
)

PHASES = {
    ord("1"): ("A", "OPEN / NEUTRAL", 5.0),
    ord("2"): ("B", "SLOW BLINK", 4.0),
    ord("3"): ("C", "BILATERAL CLOSURE", 4.0),
    ord("4"): ("D", "MOUTH NEUTRAL", 5.0),
    ord("5"): ("E", "SMILE", 3.0),
    ord("6"): ("F", "TALKING", 3.0),
    ord("7"): ("G", "SUSTAINED YAWN", 4.0),
}


def new_measurement_range():
    return {
        "left_min": None,
        "left_max": None,
        "right_min": None,
        "right_max": None,
        "mouth_min": None,
        "mouth_max": None,
        "bilateral_min": None,
        "samples": 0,
        "bilateral_closed_detected": False,
        "maximum_closure_duration": 0.0,
        "blink_events": 0,
        "yawn_events": 0,
    }


def update_measurement_range(measurements, left_ear, right_ear, mouth_ratio):
    for name, value in (
        ("left", left_ear),
        ("right", right_ear),
        ("mouth", mouth_ratio),
    ):
        if value is None or not np.isfinite(value):
            continue
        minimum_key = f"{name}_min"
        maximum_key = f"{name}_max"
        measurements[minimum_key] = (
            float(value)
            if measurements[minimum_key] is None
            else min(measurements[minimum_key], float(value))
        )
        measurements[maximum_key] = (
            float(value)
            if measurements[maximum_key] is None
            else max(measurements[maximum_key], float(value))
        )
    bilateral_value = max(left_ear, right_ear)
    measurements["bilateral_min"] = (
        float(bilateral_value)
        if measurements["bilateral_min"] is None
        else min(measurements["bilateral_min"], float(bilateral_value))
    )
    measurements["samples"] += 1


def format_range(measurements, name):
    minimum = measurements[f"{name}_min"]
    maximum = measurements[f"{name}_max"]
    if minimum is None or maximum is None:
        return "n/a"
    return f"{minimum:.3f}-{maximum:.3f}"


def print_phase_summary(phase_code, phase_name, measurements, eyes, mouth):
    print(
        f"\nPHASE {phase_code} COMPLETE — {phase_name}\n"
        f"Valid samples: {measurements['samples']}\n"
        f"Left EAR: {format_range(measurements, 'left')}\n"
        f"Right EAR: {format_range(measurements, 'right')}\n"
        f"Mouth ratio: {format_range(measurements, 'mouth')}\n"
        f"Lowest bilateral EAR: {concise_measurement(measurements['bilateral_min'])}\n"
        f"Highest mouth ratio: {concise_measurement(measurements['mouth_max'])}\n"
        f"Current eye thresholds — open L/R: "
        f"{eyes.open_threshold_left:.3f}/{eyes.open_threshold_right:.3f}; "
        f"closed L/R: {eyes.closed_threshold_left:.3f}/{eyes.closed_threshold_right:.3f}\n"
        f"Current mouth thresholds — baseline/open/release: "
        f"{concise_measurement(mouth.normal_baseline)}/"
        f"{mouth.open_threshold:.3f}/{mouth.release_threshold:.3f}"
    )
    if phase_code in ("B", "C"):
        print(
            f"Blink event detected: {'YES' if measurements['blink_events'] else 'NO'}\n"
            f"Blink count: {eyes.blink_counter}\n"
            f"Minimum left EAR: {concise_measurement(measurements['left_min'])}\n"
            f"Minimum right EAR: {concise_measurement(measurements['right_min'])}\n"
            f"Bilateral CLOSED detected: "
            f"{'YES' if measurements['bilateral_closed_detected'] else 'NO'}\n"
            f"Maximum closure duration: "
            f"{measurements['maximum_closure_duration']:.3f}s"
        )
    if phase_code in ("E", "F", "G"):
        print(
            f"Yawn event detected: {'YES' if measurements['yawn_events'] else 'NO'}\n"
            f"Yawn count: {mouth.yawn_count}\n"
            f"Maximum mouth ratio: {concise_measurement(measurements['mouth_max'])}"
        )


def draw_text(image, text, x, y, scale=0.5, color=TEXT, thickness=1):
    cv2.putText(
        image,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def select_driver(faces, frame_shape, previous_face):
    if not faces:
        return None

    frame_height, frame_width = frame_shape[:2]
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0

    def iou(first, second):
        left = max(first["x"], second["x"])
        top = max(first["y"], second["y"])
        right = min(first["x"] + first["width"], second["x"] + second["width"])
        bottom = min(first["y"] + first["height"], second["y"] + second["height"])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = first["width"] * first["height"]
        second_area = second["width"] * second["height"]
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    best = None
    best_score = float("-inf")
    for face in faces:
        center_x = face["x"] + face["width"] / 2.0
        center_y = face["y"] + face["height"] / 2.0
        distance = abs(center_x - frame_center_x) + abs(center_y - frame_center_y) * 0.6
        area = face["width"] * face["height"]
        score = area - distance * 0.35
        if previous_face is not None:
            score += iou(face, previous_face) * area * 0.75
        if score > best_score:
            best = face
            best_score = score
    return best


def combined_eye_state(left_state, right_state):
    if left_state == "OPEN" and right_state == "OPEN":
        return "OPEN"
    if left_state == "CLOSED" and right_state == "CLOSED":
        return "CLOSED"
    if "UNCERTAIN" in (left_state, right_state):
        return "UNCERTAIN"
    if "CLOSED" in (left_state, right_state):
        return "UNCERTAIN (ASYMMETRIC)"
    return "CLOSING"


def concise_measurement(value, digits=3):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def draw_points(frame, landmarks):
    if landmarks is None:
        return
    points = np.asarray(landmarks, dtype=np.int32)
    for index, (x, y) in enumerate(points):
        if 36 <= index <= 47:
            color = (80, 245, 255)
            radius = 2
        elif 48 <= index <= 67:
            color = (255, 190, 80)
            radius = 2
        else:
            color = (190, 190, 190)
            radius = 1
        cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    for eye_range in (range(36, 42), range(42, 48)):
        eye_points = points[list(eye_range)].reshape((-1, 1, 2))
        cv2.polylines(frame, [eye_points], True, (80, 245, 255), 1, cv2.LINE_AA)
    mouth_points = points[48:68].reshape((-1, 1, 2))
    cv2.polylines(frame, [mouth_points], True, (255, 190, 80), 1, cv2.LINE_AA)


def render_debug_view(
    camera_frame,
    panel_lines,
    landmarks,
    show_points,
    face_status,
    bilateral_low=False,
):
    output_width, output_height = OUTPUT_SIZE
    camera_width = output_width - PANEL_WIDTH
    source_height, source_width = camera_frame.shape[:2]
    scale = min(camera_width / source_width, output_height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(
        camera_frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    canvas[:, :camera_width] = (13, 17, 22)
    x_offset = (camera_width - resized_width) // 2
    y_offset = (output_height - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    canvas[:, camera_width:] = PANEL_BACKGROUND
    cv2.line(canvas, (camera_width, 0), (camera_width, output_height), (62, 76, 91), 2)

    draw_text(canvas, "LIVE WEBCAM — MANUAL VALIDATION", 22, 34, 0.68, TEXT, 2)
    draw_text(canvas, f"FACE: {face_status}", 22, 65, 0.55, GREEN if face_status == "DETECTED" else AMBER, 2)
    if show_points:
        draw_text(canvas, "LANDMARK OVERLAY: ON  (P toggles)", 22, 91, 0.45, BLUE, 1)
    else:
        draw_text(canvas, "LANDMARK OVERLAY: OFF  (P toggles)", 22, 91, 0.45, MUTED, 1)
    if bilateral_low:
        draw_text(canvas, "BILATERAL EAR AT SESSION LOW", 22, 120, 0.55, GREEN, 2)

    panel_x = camera_width + 18
    inner_width = PANEL_WIDTH - 36
    draw_text(canvas, "DRIVER BEHAVIOR DIAGNOSTICS", panel_x, 32, 0.58, TEXT, 2)
    y = 58

    for title, rows in panel_lines:
        card_height = 24 + len(rows) * 17
        cv2.rectangle(
            canvas,
            (panel_x, y),
            (panel_x + inner_width, y + card_height),
            PANEL_CARD,
            -1,
        )
        cv2.rectangle(
            canvas,
            (panel_x, y),
            (panel_x + inner_width, y + card_height),
            (57, 70, 85),
            1,
        )
        draw_text(canvas, title, panel_x + 11, y + 18, 0.39, MUTED, 1)
        row_y = y + 36
        for label, value, color in rows:
            draw_text(canvas, label, panel_x + 11, row_y, 0.40, MUTED, 1)
            draw_text(canvas, value, panel_x + 198, row_y, 0.39, color or TEXT, 1)
            row_y += 17
        y += card_height + 7

    instructions_top = output_height - 210
    cv2.rectangle(
        canvas,
        (panel_x, instructions_top),
        (panel_x + inner_width, output_height - 15),
        (29, 37, 47),
        -1,
    )
    draw_text(canvas, "MANUAL TESTS", panel_x + 10, instructions_top + 21, 0.45, BLUE, 1)
    for index, instruction in enumerate(TEST_INSTRUCTIONS):
        draw_text(
            canvas,
            instruction,
            panel_x + 10,
            instructions_top + 43 + index * 19,
            0.36,
            TEXT,
            1,
        )
    draw_text(canvas, "1-7: capture phase   Q: quit   R: recalibrate   P: points", panel_x + 10, output_height - 23, 0.32, MUTED, 1)

    if show_points and landmarks is not None:
        # Map source-frame landmark coordinates into the letterboxed camera region.
        mapped = np.asarray(landmarks, dtype=np.float64).copy()
        mapped[:, 0] = mapped[:, 0] * scale + x_offset
        mapped[:, 1] = mapped[:, 1] * scale + y_offset
        draw_points(canvas, mapped)

    return canvas


def main():
    print("Starting real webcam behavior validation.")
    detector = YuNetFaceDetector(
        config.MODEL_YUNET,
        config.FACE_CONFIDENCE_THRESHOLD,
        config.FACE_NMS_THRESHOLD,
    )
    predictor = LandmarkDetector(config.MODEL_DLIB)
    eyes = EyeAnalyzer()
    mouth = MouthAnalyzer()
    head_pose = HeadPoseEstimator()
    engine = DrowsinessEngine()
    alert = AlertSystem()

    camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
    if not camera.isOpened():
        print("WEBCAM: FAIL — unable to open configured camera.")
        return 1

    success, first_frame = camera.read()
    if not success or first_frame is None:
        print("WEBCAM: FAIL — no frame received.")
        camera.release()
        return 1

    print(f"WEBCAM: OPEN ({first_frame.shape[1]} x {first_frame.shape[0]})")
    print("YUNET: LOADED")
    print("DLIB LANDMARK MODEL: LOADED")
    print(
        "MEASUREMENT FORMULAS: "
        "EAR=(d(1,5)+d(2,4))/(2*d(0,3)) on eye indices 36-41 and 42-47; "
        "mouth ratio=(d(51,57)+d(53,55))/(2*d(48,54))"
    )
    print(
        "STATE BOUNDS: OPEN when value >= open threshold; "
        "CLOSED when value <= closed threshold; otherwise CLOSING. "
        "There is no separate numeric CLOSING threshold."
    )
    print(
        f"CALIBRATION GATE: at least {config.MOUTH_CALIBRATION_MIN_SAMPLES} valid "
        "normal-mouth samples and valid eye calibration are required before phase keys unlock."
    )
    print(
        "PHASE CAPTURE: press 1-7 in the diagnostic window to start the matching "
        "fixed-duration capture; wait for PHASE COMPLETE before starting the next."
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *OUTPUT_SIZE)
    previous_face = None
    last_detection_time = 0.0
    faces = []
    calibrating = True
    calibration_started = None
    calibration_failed = False
    show_points = True
    last_print_time = 0.0
    previous_frame_time = None
    camera_fps = 0.0
    latest_observation = {}
    previous_both_state = "UNCERTAIN"
    session_measurements = new_measurement_range()
    active_phase = None
    phase_measurements = None
    phase_started = None
    phase_duration = 0.0
    phase_results = {}
    bilateral_low = False
    calibration_wait_message = None

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("WEBCAM: FAIL — frame acquisition failed.")
                break

            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            if previous_frame_time is not None:
                frame_interval = now - previous_frame_time
                if frame_interval > 0:
                    instantaneous_fps = 1.0 / frame_interval
                    camera_fps = (
                        instantaneous_fps
                        if camera_fps == 0.0
                        else camera_fps * 0.85 + instantaneous_fps * 0.15
                    )
            previous_frame_time = now
            if now - last_detection_time >= config.FACE_DETECTION_INTERVAL:
                faces = detector.detect_faces(frame)
                last_detection_time = now

            face = select_driver(faces, frame.shape, previous_face)
            if face is not None:
                previous_face = face.copy()
            else:
                previous_face = None
            selected_index = faces.index(face) if face is not None else None
            detector.draw_faces(frame, faces, selected_index)

            landmarks = predictor.detect(frame, face) if face is not None else None
            landmarks_valid = DriverDrowsinessApp.landmarks_are_valid(landmarks, face, frame.shape)
            if show_points and landmarks_valid:
                draw_points(frame, landmarks)

            eye_result = eyes.analyze(landmarks if landmarks_valid and not calibrating else None, now)
            mouth_result = mouth.update(landmarks if landmarks_valid and not calibrating else None, now)
            head_result = head_pose.estimate(landmarks if landmarks_valid and not calibrating else None, frame.shape)

            if calibrating:
                if not landmarks_valid:
                    calibration_started = None
                    calibration_failed = False
                else:
                    if calibration_started is None:
                        calibration_started = now
                    eyes.add_calibration_sample(landmarks)
                    mouth.add_calibration_sample(landmarks)
                    elapsed = now - calibration_started
                    if elapsed >= config.CALIBRATION_TIME:
                        mouth_sample_count = len(mouth.calibration_samples)
                        if mouth_sample_count < config.MOUTH_CALIBRATION_MIN_SAMPLES:
                            calibration_failed = True
                            calibration_started = now
                            wait_message = (
                                f"CALIBRATION WAITING: normal-mouth samples "
                                f"{mouth_sample_count}/{config.MOUTH_CALIBRATION_MIN_SAMPLES}; "
                                "keep your mouth relaxed and face centered."
                            )
                            if wait_message != calibration_wait_message:
                                print(wait_message)
                                calibration_wait_message = wait_message
                        elif not eyes.finalize_calibration():
                            calibration_failed = True
                            calibration_started = now
                            print(
                                "CALIBRATION WAITING: insufficient valid open-eye samples; "
                                "continue looking at the camera."
                            )
                            calibration_wait_message = None
                        elif not mouth.finalize_calibration():
                            calibration_failed = True
                            calibration_started = now
                            print(
                                "CALIBRATION WAITING: collected mouth samples could not be "
                                "finalized; keep your mouth relaxed and face centered."
                            )
                            calibration_wait_message = None
                        else:
                            calibrating = False
                            calibration_failed = False
                            calibration_wait_message = None
                            print(
                                "CALIBRATION: COMPLETE | "
                                f"baseline L={eyes.baseline_left:.4f}, R={eyes.baseline_right:.4f} | "
                                f"open thresholds L={eyes.open_threshold_left:.4f}, "
                                f"R={eyes.open_threshold_right:.4f} | "
                                f"closed thresholds L={eyes.closed_threshold_left:.4f}, "
                                f"R={eyes.closed_threshold_right:.4f}"
                            )
                            if mouth.calibrated:
                                print(
                                    "MOUTH CALIBRATION: COMPLETE | "
                                    f"normal baseline={mouth.normal_baseline:.4f} | "
                                    f"open threshold={mouth.open_threshold:.4f} | "
                                    f"release threshold={mouth.release_threshold:.4f} | "
                                    f"samples={len(mouth.calibration_samples)}"
                                )
            else:
                if not landmarks_valid:
                    eye_result = eyes.analyze(None, now)
                    mouth_result = mouth.update(None, now)
                    head_result = head_pose.estimate(None, frame.shape)

                eye_state = eye_result["state"]
                head_state = head_result["state"]
                engine_output = engine.update(
                    eye_state=eye_state,
                    eye_closure_duration=eye_result["closure_duration"],
                    blink_event=eye_result["blink_detected"],
                    perclos=None,
                    head_state=head_state,
                    yawn_detected=mouth_result["yawn_detected"],
                    yawn_duration=mouth_result["yawn_duration"],
                    landmark_valid=landmarks_valid,
                    face_valid=face is not None,
                    timestamp=now,
                )
                if engine_output["alarm"] and not alert.is_alarm_active():
                    alert.start_alarm()
                elif not engine_output["alarm"] and alert.is_alarm_active():
                    alert.stop_alarm()
                alert.update(now)

                current_both_state = (
                    combined_eye_state(eyes.left_state, eyes.right_state)
                    if landmarks_valid
                    else "UNCERTAIN"
                )
                if current_both_state != previous_both_state:
                    print(
                        f"EYE TRANSITION: {previous_both_state} -> {current_both_state} | "
                        f"L={concise_measurement(eye_result['left_ear'])} "
                        f"R={concise_measurement(eye_result['right_ear'])} | "
                        f"closure={eye_result['closure_duration']:.3f}s"
                    )
                    previous_both_state = current_both_state

                if mouth_result.get("state_transition"):
                    print(
                        f"MOUTH TRANSITION: {mouth_result['state_transition']} | "
                        f"ratio={concise_measurement(mouth_result['mouth_ratio'])} | "
                        f"duration={mouth_result['yawn_duration']:.3f}s"
                    )
                if eye_result["blink_detected"]:
                    print(
                        f"BLINK EVENT: TRUE | count={eyes.blink_counter} | "
                        f"closure={eye_result['blink_duration']:.3f}s"
                    )
                if mouth_result["yawn_detected"]:
                    print(
                        f"YAWN EVENT: TRUE | count={mouth.yawn_count} | "
                        f"duration={mouth_result['yawn_duration']:.3f}s"
                    )
                if now - last_print_time >= 1.0:
                    last_print_time = now
                    print(
                        f"FACE={'DETECTED' if face else 'LOST'} "
                        f"LANDMARKS={'VALID' if landmarks_valid else 'INVALID'} "
                        f"LEFT_EYE={concise_measurement(eye_result['left_ear'])}/"
                        f"{concise_measurement(eyes.open_threshold_left)}/"
                        f"{concise_measurement(eyes.closed_threshold_left)} "
                        f"RIGHT_EYE={concise_measurement(eye_result['right_ear'])}/"
                        f"{concise_measurement(eyes.open_threshold_right)}/"
                        f"{concise_measurement(eyes.closed_threshold_right)} "
                        f"EYE_STATE={combined_eye_state(eyes.left_state, eyes.right_state)} "
                        f"CLOSURE={eye_result['closure_duration']:.2f}s "
                        f"BLINK={eye_result['blink_detected']} "
                        f"BLINK_COUNT={eyes.blink_counter} "
                        f"MOUTH={concise_measurement(mouth_result['mouth_ratio'])}/"
                        f"{concise_measurement(mouth.normal_baseline)}/"
                        f"{concise_measurement(mouth.open_threshold)}/"
                        f"{concise_measurement(mouth.release_threshold)} "
                        f"MOUTH_STATE={mouth_result['state']} "
                        f"YAWN_DURATION={mouth_result['yawn_duration']:.2f}s "
                        f"YAWN_EVENT={mouth_result['yawn_detected']} "
                        f"YAWN_COUNT={mouth.yawn_count} "
                        f"PERCLOS={engine_output['perclos']:.3f} "
                        f"SCORE={engine_output['score']:.1f} "
                        f"ENGINE={engine_output['reason']} "
                        f"EYE_MACHINE={eye_result['state_machine_state']}"
                    )
                latest_observation = {
                    "engine": engine_output,
                    "eye": eye_result,
                    "mouth": mouth_result,
                    "head": head_result,
                }

                if landmarks_valid:
                    left_ear = eye_result["left_ear"]
                    right_ear = eye_result["right_ear"]
                    mouth_ratio = mouth_result["mouth_ratio"]
                    update_measurement_range(
                        session_measurements,
                        left_ear,
                        right_ear,
                        mouth_ratio,
                    )
                    if active_phase is not None:
                        update_measurement_range(
                            phase_measurements,
                            left_ear,
                            right_ear,
                            mouth_ratio,
                        )
                        phase_measurements["bilateral_closed_detected"] |= (
                            eyes.left_state == "CLOSED"
                            and eyes.right_state == "CLOSED"
                        )
                        phase_measurements["maximum_closure_duration"] = max(
                            phase_measurements["maximum_closure_duration"],
                            float(eye_result["closure_duration"]),
                        )
                        phase_measurements["blink_events"] += int(
                            eye_result["blink_detected"]
                        )
                        phase_measurements["yawn_events"] += int(
                            mouth_result["yawn_detected"]
                        )
                    bilateral_low = (
                        left_ear <= session_measurements["left_min"] + 0.015
                        and right_ear <= session_measurements["right_min"] + 0.015
                    )
                else:
                    bilateral_low = False

                if (
                    active_phase is not None
                    and now - phase_started >= phase_duration
                ):
                    phase_code, phase_name, _ = active_phase
                    phase_results[phase_code] = phase_measurements
                    print_phase_summary(
                        phase_code,
                        phase_name,
                        phase_measurements,
                        eyes,
                        mouth,
                    )
                    active_phase = None
                    phase_measurements = None
                    phase_started = None

            left_state = eyes.left_state if not calibrating and landmarks_valid else "UNCERTAIN"
            right_state = eyes.right_state if not calibrating and landmarks_valid else "UNCERTAIN"
            both_state = (
                combined_eye_state(left_state, right_state)
                if not calibrating and landmarks_valid
                else "UNCERTAIN"
            )
            current = latest_observation
            current_eye = current.get("eye", eye_result)
            current_mouth = current.get("mouth", mouth_result)
            current_head = current.get("head", head_result)
            current_engine = current.get("engine", {
                "perclos": 0.0,
                "score": engine.score,
                "status": engine.status,
                "reason": "CALIBRATING" if calibrating else "UNCERTAIN",
                "score_components": engine.score_components,
            })
            if calibrating:
                current_eye = eye_result
                current_mouth = mouth_result
                current_head = head_result

            calibration_label = (
                "CALIBRATING — KEEP EYES OPEN"
                if calibrating and not calibration_failed
                else "CALIBRATION NEEDS MORE OPEN SAMPLES"
                if calibrating
                else "READY"
            )
            closure_duration = current_eye.get("closure_duration", 0.0) or 0.0
            phase_label = (
                f"{active_phase[0]} {max(0.0, now - phase_started):.1f}/{phase_duration:.0f}s"
                if active_phase is not None
                else "READY"
            )
            panel_lines = [
                ("PERCEPTION", [
                    ("FACE", "DETECTED" if face is not None else "LOST", GREEN if face is not None else AMBER),
                    ("LANDMARKS", "VALID" if landmarks_valid else "INVALID", GREEN if landmarks_valid else AMBER),
                    ("FRAME / FPS", f"{frame.shape[1]}x{frame.shape[0]} / {camera_fps:.1f}", TEXT),
                    ("CALIBRATION", calibration_label, AMBER if calibrating else GREEN),
                    (
                        "MOUTH SAMPLES",
                        f"{len(mouth.calibration_samples)}/{config.MOUTH_CALIBRATION_MIN_SAMPLES}",
                        AMBER if len(mouth.calibration_samples) < config.MOUTH_CALIBRATION_MIN_SAMPLES else GREEN,
                    ),
                ]),
                ("MEASUREMENT CAPTURE", [
                    ("ACTIVE PHASE", phase_label, AMBER if active_phase else GREEN),
                    ("SESSION LEFT MIN-MAX", format_range(session_measurements, "left"), TEXT),
                    ("SESSION RIGHT MIN-MAX", format_range(session_measurements, "right"), TEXT),
                    ("SESSION BILATERAL LOW", concise_measurement(session_measurements["bilateral_min"]), TEXT),
                ]),
                ("EYE MONITOR", [
                    ("LEFT EYE", left_state, GREEN if left_state == "OPEN" else AMBER),
                    ("RIGHT EYE", right_state, GREEN if right_state == "OPEN" else AMBER),
                    ("BOTH EYES", both_state, GREEN if both_state == "OPEN" else AMBER),
                    ("LEFT MEASUREMENT", concise_measurement(current_eye.get("left_ear")), TEXT),
                    ("RIGHT MEASUREMENT", concise_measurement(current_eye.get("right_ear")), TEXT),
                    ("BASELINE L / R", f"{concise_measurement(eyes.baseline_left)} / {concise_measurement(eyes.baseline_right)}", MUTED),
                    ("LEFT OPEN THRESHOLD", concise_measurement(eyes.open_threshold_left), MUTED),
                    ("LEFT CLOSED THRESHOLD", concise_measurement(eyes.closed_threshold_left), MUTED),
                    ("RIGHT OPEN THRESHOLD", concise_measurement(eyes.open_threshold_right), MUTED),
                    ("RIGHT CLOSED THRESHOLD", concise_measurement(eyes.closed_threshold_right), MUTED),
                    ("CLOSURE DURATION", f"{closure_duration:.2f} s", TEXT),
                    ("BLINK EVENT", "YES" if current_eye.get("blink_detected") else "NO", GREEN if current_eye.get("blink_detected") else MUTED),
                    ("BLINK COUNT", str(eyes.blink_counter), TEXT),
                    ("STATE MACHINE", current_eye.get("state_machine_state", "UNCERTAIN"), BLUE),
                ]),
                ("MOUTH / HEAD", [
                    ("MOUTH", current_mouth.get("state", "MOUTH UNCERTAIN"), TEXT),
                    ("MOUTH MEASUREMENT", concise_measurement(current_mouth.get("mouth_ratio")), TEXT),
                    ("NORMAL BASELINE", concise_measurement(mouth.normal_baseline), MUTED),
                    ("OPEN / RELEASE THRESHOLD", f"{mouth.open_threshold:.3f} / {mouth.release_threshold:.3f}", MUTED),
                    ("YAWN DURATION", f"{current_mouth.get('yawn_duration', 0.0):.2f} / {config.YAWN_CONFIRM_TIME:.1f} s", TEXT),
                    ("YAWN EVENT / COUNT", f"{'YES' if current_mouth.get('yawn_detected') else 'NO'} / {mouth.yawn_count}", TEXT),
                    ("HEAD", current_head.get("state", "UNCERTAIN"), TEXT),
                ]),
                ("TEMPORAL ENGINE", [
                    ("PERCLOS", f"{current_engine.get('perclos', 0.0):.3f}", TEXT),
                    ("SCORE", f"{current_engine.get('score', engine.score):.1f} / 100", TEXT),
                    ("STATUS / ALARM", f"{current_engine.get('status', engine.status)} / {'ACTIVE' if alert.is_alarm_active() else 'OFF'}", RED if alert.is_alarm_active() else GREEN),
                    ("ENGINE REASON", current_engine.get("reason", "UNCERTAIN"), MUTED),
                ]),
            ]

            view = render_debug_view(
                frame,
                panel_lines,
                landmarks if landmarks_valid else None,
                show_points,
                "DETECTED" if face is not None else "LOST",
                bilateral_low,
            )
            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                show_points = not show_points
            if key in PHASES:
                if calibrating:
                    print("PHASE CAPTURE: wait until calibration is complete.")
                elif active_phase is not None:
                    print(
                        f"PHASE CAPTURE: finish phase {active_phase[0]} before starting another."
                    )
                else:
                    active_phase = PHASES[key]
                    phase_code, phase_name, phase_duration = active_phase
                    phase_started = time.monotonic()
                    phase_measurements = new_measurement_range()
                    print(
                        f"PHASE {phase_code} ({phase_name}) STARTED for {phase_duration:.0f}s | "
                        f"OPEN L/R={eyes.open_threshold_left:.3f}/{eyes.open_threshold_right:.3f} | "
                        f"CLOSED L/R={eyes.closed_threshold_left:.3f}/{eyes.closed_threshold_right:.3f} | "
                        f"MOUTH baseline/open/release="
                        f"{concise_measurement(mouth.normal_baseline)}/"
                        f"{mouth.open_threshold:.3f}/{mouth.release_threshold:.3f}"
                    )
            if key == ord("r"):
                eyes.reset_calibration()
                mouth.reset_calibration()
                mouth.reset_tracking()
                calibrating = True
                calibration_started = None
                calibration_failed = False
                latest_observation = {}
                if active_phase is not None:
                    print(f"PHASE {active_phase[0]} CANCELLED: recalibration requested.")
                active_phase = None
                phase_measurements = None
                phase_started = None
                print("CALIBRATION: reset; keep both eyes open and face the camera.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        alert.stop_alarm()

    print("Manual validation window closed.")
    print(
        f"SESSION EXTREMA | valid_samples={session_measurements['samples']} | "
        f"LEFT_EAR={format_range(session_measurements, 'left')} | "
        f"RIGHT_EAR={format_range(session_measurements, 'right')} | "
        f"MOUTH={format_range(session_measurements, 'mouth')} | "
        f"LOWEST_BILATERAL_EAR(max(L,R))="
        f"{concise_measurement(session_measurements['bilateral_min'])}"
    )
    print("COMPLETED PHASE SUMMARY")
    print("PHASE | VALID SAMPLES | LEFT EAR MIN-MAX | RIGHT EAR MIN-MAX | MOUTH MIN-MAX")
    for phase_code in ("A", "B", "C", "D", "E", "F", "G"):
        measurements = phase_results.get(phase_code)
        if measurements is not None:
            print(
                f"{phase_code} | {measurements['samples']} | "
                f"{format_range(measurements, 'left')} | "
                f"{format_range(measurements, 'right')} | "
                f"{format_range(measurements, 'mouth')}"
            )
    print("No blink/yawn/closure behavior is claimed verified unless you observed and performed those tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

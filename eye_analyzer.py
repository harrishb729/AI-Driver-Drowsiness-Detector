import time

import numpy as np

import config
from utils import eye_aspect_ratio

LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))


class EyeAnalyzer:
    def __init__(self):
        self.calibration_samples = []
        self.baseline_left = None
        self.baseline_right = None
        self.open_threshold_left = config.EYE_OPEN_THRESHOLD
        self.open_threshold_right = config.EYE_OPEN_THRESHOLD
        self.closed_threshold_left = config.EYE_CLOSED_THRESHOLD
        self.closed_threshold_right = config.EYE_CLOSED_THRESHOLD
        self.calibrated = False
        self.calibration_started = time.monotonic()
        self.last_state = "UNCERTAIN"
        self.current_state = "UNCERTAIN"
        self.state_machine_state = "UNCERTAIN"
        self.left_state = "UNCERTAIN"
        self.right_state = "UNCERTAIN"
        self.blink_count = 0
        self.blink_state = "UNCERTAIN"
        self.blink_candidate_started = None
        self.blink_candidate_has_evidence = False
        self.blink_candidate_expired = False
        self.last_blink_time = None
        self.closure_started = None
        self.closure_saw_bilateral_closed = False
        self.closure_interrupted_by_uncertainty = False
        self.last_valid_time = None
        self.uncertain_started = None
        self.sustained_closure_duration = 0.0
        self.last_result = self._result()

    def _result(
        self,
        state="EYES UNCERTAIN",
        left_ear=None,
        right_ear=None,
        blink_detected=False,
        blink_duration=None,
        blink_candidate_duration=0.0,
        closure_duration=0.0,
        tracking_valid=False,
        state_transition=None,
    ):
        return {
            "state": state,
            "state_machine_state": self.state_machine_state,
            "state_transition": state_transition,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "blink_detected": blink_detected,
            "blink_state": self.blink_state,
            "blink_duration": (
                None if blink_duration is None else float(blink_duration)
            ),
            "blink_candidate_duration": float(blink_candidate_duration),
            "blink_count": self.blink_count,
            "closure_duration": float(closure_duration),
            "sustained_closure_duration": float(closure_duration),
            "sustained_closure_active": (
                self.state_machine_state == "SUSTAINED_CLOSURE"
            ),
            "tracking_valid": tracking_valid,
        }

    def reset_calibration(self):
        self.calibration_samples = []
        self.baseline_left = None
        self.baseline_right = None
        self.open_threshold_left = config.EYE_OPEN_THRESHOLD
        self.open_threshold_right = config.EYE_OPEN_THRESHOLD
        self.closed_threshold_left = config.EYE_CLOSED_THRESHOLD
        self.closed_threshold_right = config.EYE_CLOSED_THRESHOLD
        self.calibrated = False
        self.calibration_started = time.monotonic()
        self.reset_tracking()

    def reset_tracking(self):
        self.last_state = "UNCERTAIN"
        self.current_state = "UNCERTAIN"
        self.state_machine_state = "UNCERTAIN"
        self.left_state = "UNCERTAIN"
        self.right_state = "UNCERTAIN"
        self.closure_started = None
        self.closure_saw_bilateral_closed = False
        self.closure_interrupted_by_uncertainty = False
        self.blink_state = "UNCERTAIN"
        self.blink_candidate_started = None
        self.blink_candidate_has_evidence = False
        self.blink_candidate_expired = False
        self.last_blink_time = None
        self.last_valid_time = None
        self.uncertain_started = None
        self.sustained_closure_duration = 0.0

    @staticmethod
    def _measure(landmarks):
        if landmarks is None:
            return None, None
        points = np.asarray(landmarks, dtype=np.float64)
        if points.shape != (68, 2) or not np.isfinite(points).all():
            return None, None
        return (
            eye_aspect_ratio(points, LEFT_EYE_INDICES),
            eye_aspect_ratio(points, RIGHT_EYE_INDICES),
        )

    def add_calibration_sample(self, landmarks):
        left_ear, right_ear = self._measure(landmarks)
        if left_ear is None or right_ear is None:
            return False
        if (
            left_ear < config.EYE_OPEN_THRESHOLD
            or right_ear < config.EYE_OPEN_THRESHOLD
            or left_ear > 0.75
            or right_ear > 0.75
        ):
            return False
        self.calibration_samples.append((float(left_ear), float(right_ear)))
        return True

    def finalize_calibration(self):
        if len(self.calibration_samples) < config.CALIBRATION_MIN_SAMPLES:
            self.calibrated = False
            return False

        samples = np.asarray(self.calibration_samples, dtype=np.float64)
        self.baseline_left = float(np.median(samples[:, 0]))
        self.baseline_right = float(np.median(samples[:, 1]))
        self.open_threshold_left = max(
            config.EYE_OPEN_THRESHOLD,
            self.baseline_left * config.EYE_OPEN_BASELINE_RATIO,
        )
        self.open_threshold_right = max(
            config.EYE_OPEN_THRESHOLD,
            self.baseline_right * config.EYE_OPEN_BASELINE_RATIO,
        )
        self.closed_threshold_left = min(
            config.EYE_CLOSED_THRESHOLD,
            self.baseline_left * config.EYE_CLOSED_BASELINE_RATIO,
        )
        self.closed_threshold_right = min(
            config.EYE_CLOSED_THRESHOLD,
            self.baseline_right * config.EYE_CLOSED_BASELINE_RATIO,
        )
        self.calibrated = True
        self.reset_tracking()
        return True

    @staticmethod
    def _classify_eye(value, previous_state, open_threshold, closed_threshold):
        if value is None or not np.isfinite(value):
            return "UNCERTAIN"
        if value >= open_threshold:
            return "OPEN"
        if value <= closed_threshold:
            return "CLOSED"
        if previous_state == "CLOSED":
            return "CLOSED"
        return "CLOSING"

    def _uncertain(self, now, left_ear=None, right_ear=None):
        previous_blink_state = self.blink_state
        self.blink_state = "UNCERTAIN"
        self.blink_candidate_started = None
        self.blink_candidate_has_evidence = False
        self.blink_candidate_expired = False
        previous_state = self.state_machine_state
        self.current_state = "EYES UNCERTAIN"
        self.state_machine_state = "UNCERTAIN"
        if self.uncertain_started is None:
            self.uncertain_started = now
        if self.closure_started is not None:
            self.closure_interrupted_by_uncertainty = True
        if (
            self.closure_started is not None
            and now - self.uncertain_started > config.EYE_UNCERTAIN_GRACE_TIME
        ):
            self.closure_started = None
            self.closure_saw_bilateral_closed = False
            self.closure_interrupted_by_uncertainty = False
            self.sustained_closure_duration = 0.0
            self.left_state = "UNCERTAIN"
            self.right_state = "UNCERTAIN"
        result = self._result(
            state=self.current_state,
            left_ear=left_ear,
            right_ear=right_ear,
            closure_duration=self.sustained_closure_duration,
            state_transition=(
                f"{previous_state} -> {self.state_machine_state}"
                if previous_state != self.state_machine_state
                else None
            ),
        )
        self.last_result = result
        if config.DEBUG_MODE and previous_blink_state != self.blink_state:
            self._log_blink(result)
        return result

    def _log_blink(self, result):
        print(
            "BLINK: "
            f"state={self.blink_state} "
            f"left_ear={result['left_ear']} "
            f"right_ear={result['right_ear']} "
            f"candidate_duration={result['blink_candidate_duration']:.3f} "
            f"event_count={self.blink_count}"
        )

    def _update_blink_event(
        self,
        now,
        left_ear,
        right_ear,
        left_state,
        right_state,
    ):
        previous_state = self.blink_state
        both_open = left_state == "OPEN" and right_state == "OPEN"
        both_closed = left_state == "CLOSED" and right_state == "CLOSED"
        candidate_reduction = (
            left_ear
            <= self.baseline_left * config.BLINK_CANDIDATE_BASELINE_RATIO
            and right_ear
            <= self.baseline_right * config.BLINK_CANDIDATE_BASELINE_RATIO
        )
        strong_bilateral_evidence = (
            left_ear
            <= self.baseline_left * config.BLINK_EVIDENCE_BASELINE_RATIO
            and right_ear
            <= self.baseline_right * config.BLINK_EVIDENCE_BASELINE_RATIO
        )
        blink_detected = False
        blink_duration = None

        if self.blink_candidate_started is None:
            cooldown_elapsed = (
                self.last_blink_time is None
                or now - self.last_blink_time >= config.BLINK_REFRACTORY_PERIOD
            )
            if candidate_reduction and cooldown_elapsed:
                self.blink_candidate_started = now
                self.blink_candidate_has_evidence = strong_bilateral_evidence or both_closed
                self.blink_candidate_expired = False
                self.blink_state = "CLOSING"
            elif both_open:
                self.blink_state = "OPEN"
            elif both_closed:
                self.blink_state = "CLOSED"
            elif left_state == "CLOSING" and right_state == "CLOSING":
                self.blink_state = "CLOSING"
            else:
                self.blink_state = "UNCERTAIN"
        else:
            duration = max(0.0, now - self.blink_candidate_started)
            self.blink_candidate_has_evidence |= (
                strong_bilateral_evidence or both_closed
            )
            if both_open:
                blink_duration = duration
                if (
                    not self.blink_candidate_expired
                    and self.blink_candidate_has_evidence
                    and config.BLINK_MIN_DURATION <= duration <= config.BLINK_MAX_DURATION
                    and (
                        self.last_blink_time is None
                        or now - self.last_blink_time
                        >= config.BLINK_REFRACTORY_PERIOD
                    )
                ):
                    self.blink_count += 1
                    self.last_blink_time = now
                    blink_detected = True
                    self.blink_state = "RECOVERY"
                else:
                    self.blink_state = "OPEN"
                self.blink_candidate_started = None
                self.blink_candidate_has_evidence = False
                self.blink_candidate_expired = False
            elif duration > config.BLINK_MAX_DURATION:
                self.blink_candidate_expired = True
                self.blink_state = "CLOSED" if both_closed else "CLOSING"
            else:
                self.blink_state = "CLOSED" if both_closed else "CLOSING"

        if config.DEBUG_MODE and (
            previous_state != self.blink_state or blink_detected
        ):
            result = self._result(
                left_ear=left_ear,
                right_ear=right_ear,
                blink_detected=blink_detected,
                blink_candidate_duration=(
                    blink_duration
                    if blink_detected and blink_duration is not None
                    else max(0.0, now - self.blink_candidate_started)
                    if self.blink_candidate_started is not None
                    else 0.0
                ),
            )
            self._log_blink(result)
        return blink_detected, blink_duration

    def analyze(self, landmarks, now=None):
        now = time.monotonic() if now is None else float(now)
        left_ear, right_ear = self._measure(landmarks)
        if not self.calibrated or left_ear is None or right_ear is None:
            return self._uncertain(now, left_ear, right_ear)

        left_state = self._classify_eye(
            left_ear,
            self.left_state,
            self.open_threshold_left,
            self.closed_threshold_left,
        )
        right_state = self._classify_eye(
            right_ear,
            self.right_state,
            self.open_threshold_right,
            self.closed_threshold_right,
        )
        if left_state == "UNCERTAIN" or right_state == "UNCERTAIN":
            return self._uncertain(now, left_ear, right_ear)

        self.uncertain_started = None
        previous_machine_state = self.state_machine_state
        blink_detected, blink_duration = self._update_blink_event(
            now,
            left_ear,
            right_ear,
            left_state,
            right_state,
        )
        both_open = left_state == "OPEN" and right_state == "OPEN"
        both_closed = left_state == "CLOSED" and right_state == "CLOSED"
        bilateral_closing = (
            left_state in ("CLOSING", "CLOSED")
            and right_state in ("CLOSING", "CLOSED")
            and not both_open
        )

        if both_closed:
            if self.closure_started is None:
                self.closure_started = now
                self.closure_interrupted_by_uncertainty = False
            self.closure_saw_bilateral_closed = True
            self.sustained_closure_duration = max(0.0, now - self.closure_started)
            if self.sustained_closure_duration >= config.EYE_CLOSURE_CONFIRM_TIME:
                self.state_machine_state = "SUSTAINED_CLOSURE"
                self.current_state = "EYES CLOSED"
            else:
                self.state_machine_state = "CLOSURE_CANDIDATE"
                self.current_state = "EYES CLOSING"
            self.last_valid_time = now
        elif bilateral_closing:
            if self.closure_started is None:
                self.closure_started = now
                self.closure_saw_bilateral_closed = False
                self.closure_interrupted_by_uncertainty = False
            self.sustained_closure_duration = max(0.0, now - self.closure_started)
            self.state_machine_state = "CLOSURE_CANDIDATE"
            self.current_state = "EYES CLOSING"
            self.last_valid_time = now
        elif both_open:
            if self.closure_started is not None:
                closure_duration = max(0.0, now - self.closure_started)
                self.state_machine_state = "RECOVERY"
                self.sustained_closure_duration = 0.0
                self.closure_started = None
                self.closure_saw_bilateral_closed = False
                self.closure_interrupted_by_uncertainty = False
            else:
                self.state_machine_state = "OPEN"
                self.sustained_closure_duration = 0.0
            self.current_state = "EYES OPEN"
            self.last_valid_time = now
        else:
            self.current_state = "EYES UNCERTAIN"
            self.state_machine_state = "UNCERTAIN"
            self.closure_started = None
            self.closure_saw_bilateral_closed = False
            self.closure_interrupted_by_uncertainty = False
            self.sustained_closure_duration = 0.0

        self.left_state = left_state
        self.right_state = right_state
        self.last_state = self.current_state
        result = self._result(
            state=self.current_state,
            left_ear=left_ear,
            right_ear=right_ear,
            blink_detected=blink_detected,
            blink_duration=blink_duration,
            blink_candidate_duration=(
                max(0.0, now - self.blink_candidate_started)
                if self.blink_candidate_started is not None
                else 0.0
            ),
            closure_duration=self.sustained_closure_duration,
            tracking_valid=True,
            state_transition=(
                f"{previous_machine_state} -> {self.state_machine_state}"
                if previous_machine_state != self.state_machine_state
                else None
            ),
        )
        self.last_result = result
        return result

    @property
    def blink_counter(self):
        return self.blink_count

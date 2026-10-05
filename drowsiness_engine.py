from collections import deque
import time

import config
from utils import clamp


class DrowsinessEngine:
    def __init__(self):
        self.score = 0.0
        self.status = "NORMAL"
        self.alarm_active = False
        self.last_update_time = None
        self.last_valid_time = None
        self.last_valid_eye_closed = None
        self.eye_history = deque()
        self.blink_times = deque()
        self.last_yawn_time = None
        self.head_down_started = None
        self.previous_warning = False
        self.score_components = {
            "eye_closure": 0.0,
            "perclos": 0.0,
            "yawn": 0.0,
            "head_pose": 0.0,
            "blink_pattern": 0.0,
        }

    @staticmethod
    def _status_from_score(score):
        score_band = int(score)
        for status, (minimum, maximum) in config.STATUS_BANDS.items():
            if minimum <= score_band <= maximum:
                return status
        return "WARNING" if score_band > config.STATUS_BANDS["WARNING"][1] else "NORMAL"

    def _record_eye_history(self, now, eye_closed, valid):
        if not valid:
            self.last_valid_time = None
            self.last_valid_eye_closed = None
            return
        if self.last_valid_time is not None:
            interval = now - self.last_valid_time
            if 0.0 < interval <= config.EYE_VALID_GAP_SECONDS:
                self.eye_history.append(
                    (self.last_valid_time, now, bool(self.last_valid_eye_closed))
                )
        self.last_valid_time = now
        self.last_valid_eye_closed = bool(eye_closed)
        cutoff = now - config.EYE_PERCLOS_WINDOW_SECONDS
        while self.eye_history and self.eye_history[0][1] < cutoff:
            self.eye_history.popleft()

    def _perclos(self, now):
        window_start = now - config.EYE_PERCLOS_WINDOW_SECONDS
        closed_duration = 0.0
        valid_duration = 0.0
        for start, end, closed in self.eye_history:
            clipped_start = max(start, window_start)
            duration = max(0.0, end - clipped_start)
            valid_duration += duration
            if closed:
                closed_duration += duration
        perclos = clamp(closed_duration / valid_duration, 0.0, 1.0) if valid_duration else 0.0
        return perclos, valid_duration

    def _recent_blink_component(self, now):
        cutoff = now - config.BLINK_HISTORY_SECONDS
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()
        excess = max(
            0,
            len(self.blink_times) - config.BLINK_ABNORMAL_COUNT_PER_MINUTE,
        )
        return min(
            config.BLINK_PATTERN_SCORE_MAX,
            excess * config.BLINK_PATTERN_SCORE_STEP,
        )

    def update(
        self,
        eye_state="EYES UNCERTAIN",
        sustained_closure_duration=0.0,
        head_state="UNCERTAIN",
        yawn_detected=False,
        current_time=None,
        *,
        eye_closure_duration=None,
        blink_event=False,
        perclos=None,
        yawn_duration=0.0,
        head_down_duration=None,
        landmark_valid=True,
        face_valid=True,
        timestamp=None,
    ):
        now = timestamp if timestamp is not None else current_time
        now = time.monotonic() if now is None else float(now)
        dt = 0.0 if self.last_update_time is None else max(0.0, now - self.last_update_time)
        self.last_update_time = now

        closure_duration = (
            sustained_closure_duration
            if eye_closure_duration is None
            else eye_closure_duration
        )
        closure_duration = max(0.0, float(closure_duration or 0.0))
        valid = bool(face_valid and landmark_valid and eye_state != "EYES UNCERTAIN")
        eye_closed = valid and eye_state == "EYES CLOSED"
        self._record_eye_history(now, eye_closed, valid)
        measured_perclos, valid_history_seconds = self._perclos(now)
        perclos_value = (
            measured_perclos
            if perclos is None
            else clamp(float(perclos), 0.0, 1.0)
        )

        events = []
        if blink_event:
            self.blink_times.append(now)
            events.append("BLINK")
        if yawn_detected:
            self.last_yawn_time = now
            events.append("YAWN")

        closure_component = 0.0
        if valid and eye_state in ("EYES CLOSED", "EYES CLOSING"):
            closure_component = config.EYE_CLOSURE_SCORE_MAX * clamp(
                closure_duration / config.EYE_CLOSURE_WARNING_TIME,
                0.0,
                1.0,
            )
        perclos_component = (
            config.PERCLOS_SCORE_MAX
            * clamp(perclos_value / config.PERCLOS_SCORE_THRESHOLD, 0.0, 1.0)
            if (
                perclos is not None
                or valid_history_seconds >= config.PERCLOS_MIN_HISTORY_SECONDS
            )
            else 0.0
        )

        yawn_component = 0.0
        if self.last_yawn_time is not None:
            yawn_age = max(0.0, now - self.last_yawn_time)
            if yawn_age < config.YAWN_SCORE_DECAY_SECONDS:
                yawn_component = config.YAWN_SCORE_MAX * (
                    1.0 - yawn_age / config.YAWN_SCORE_DECAY_SECONDS
                )

        if valid and head_state == "DOWN":
            if self.head_down_started is None:
                self.head_down_started = now
            measured_head_duration = max(0.0, now - self.head_down_started)
            if head_down_duration is not None:
                measured_head_duration = max(measured_head_duration, float(head_down_duration))
            head_component = config.HEAD_POSE_SCORE_MAX * clamp(
                measured_head_duration / config.HEAD_DOWN_FULL_SCORE_SECONDS,
                0.0,
                1.0,
            )
        else:
            self.head_down_started = None
            head_component = 0.0

        blink_component = self._recent_blink_component(now)
        self.score_components = {
            "eye_closure": closure_component,
            "perclos": perclos_component,
            "yawn": yawn_component,
            "head_pose": head_component,
            "blink_pattern": blink_component,
        }
        target_score = clamp(sum(self.score_components.values()), 0.0, 100.0)
        if not valid:
            target_score = 0.0

        if dt > 0:
            rate = (
                config.SCORE_RISE_RATE_PER_SECOND
                if target_score > self.score
                else config.RECOVERY_RATE_PER_SECOND
            )
            delta = rate * dt
            if target_score > self.score:
                self.score = min(target_score, self.score + delta)
            else:
                self.score = max(target_score, self.score - delta)
        self.score = clamp(self.score, 0.0, 100.0)
        self.status = self._status_from_score(self.score)

        warning = self.score >= config.ALARM_SCORE
        if not self.alarm_active and warning:
            self.alarm_active = True
            events.append("WARNING")
            events.append("ALARM_START")
        elif self.alarm_active and self.score <= config.ALARM_OFF_SCORE:
            self.alarm_active = False
            events.append("ALARM_STOP")
            if self.previous_warning:
                events.append("RECOVERY")
        self.previous_warning = self.alarm_active

        return {
            "score": float(self.score),
            "target_score": float(target_score),
            "status": self.status,
            "warning": self.score >= config.ALARM_SCORE,
            "alarm": self.alarm_active,
            "perclos": float(perclos_value),
            "score_components": dict(self.score_components),
            "reason": self._primary_reason(),
            "events": events,
        }

    def _primary_reason(self):
        if self.score_components["eye_closure"] > 0:
            return "EYE CLOSURE"
        if self.score_components["perclos"] >= 10:
            return "ELEVATED EYE CLOSURE HISTORY"
        if self.score_components["head_pose"] > 0:
            return "PROLONGED HEAD DOWN"
        if self.score_components["yawn"] > 0:
            return "CONFIRMED YAWN"
        if self.score_components["blink_pattern"] > 0:
            return "ABNORMAL BLINK FREQUENCY"
        return "NORMAL"

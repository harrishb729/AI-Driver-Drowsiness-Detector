import time

import numpy as np

import config
from utils import mouth_aspect_ratio

MOUTH_INDICES = list(range(48, 68))


class MouthAnalyzer:
    def __init__(self):
        self.state = "MOUTH NORMAL"
        self.yawn_count = 0
        self.calibration_samples = []
        self.normal_baseline = None
        self.open_threshold = config.YAWN_THRESHOLD + config.MOUTH_YAWN_BASELINE_MARGIN
        self.release_threshold = self.open_threshold * config.MOUTH_OPEN_HYSTERESIS_RATIO
        self.calibrated = False
        self.opening_started = None
        self.last_valid_time = None
        self.last_yawn_time = None
        self.yawn_duration = 0.0
        self.event_counted = False
        self.confirmed_started = None
        self.recovery_started = None
        self.rearm_started = None
        self.episode_confirmed = False

    def add_calibration_sample(self, landmarks):
        ratio = mouth_aspect_ratio(landmarks, MOUTH_INDICES)
        if (
            ratio is None
            or not np.isfinite(ratio)
            or ratio > config.MOUTH_CALIBRATION_MAX_RATIO
        ):
            return False
        self.calibration_samples.append(float(ratio))
        return True

    def finalize_calibration(self):
        if len(self.calibration_samples) < config.MOUTH_CALIBRATION_MIN_SAMPLES:
            self.calibrated = False
            return False

        samples = np.asarray(self.calibration_samples, dtype=np.float64)
        lower_quartile = np.quantile(samples, 0.25)
        upper_quartile = np.quantile(samples, 0.75)
        central_samples = samples[
            (samples >= lower_quartile) & (samples <= upper_quartile)
        ]
        self.normal_baseline = float(np.median(central_samples))
        self.open_threshold = max(
            config.YAWN_THRESHOLD,
            self.normal_baseline * config.MOUTH_YAWN_BASELINE_FACTOR,
            self.normal_baseline + config.MOUTH_YAWN_BASELINE_MARGIN,
        )
        self.release_threshold = (
            self.open_threshold * config.MOUTH_OPEN_HYSTERESIS_RATIO
        )
        self.calibrated = True
        self.reset_tracking()
        return True

    def reset_calibration(self):
        self.calibration_samples = []
        self.normal_baseline = None
        self.open_threshold = config.YAWN_THRESHOLD + config.MOUTH_YAWN_BASELINE_MARGIN
        self.release_threshold = (
            self.open_threshold * config.MOUTH_OPEN_HYSTERESIS_RATIO
        )
        self.calibrated = False
        self.reset_tracking()

    def reset_tracking(self):
        self.state = "MOUTH NORMAL"
        self.opening_started = None
        self.last_valid_time = None
        self.yawn_duration = 0.0
        self.event_counted = False
        self.confirmed_started = None
        self.recovery_started = None
        self.rearm_started = None
        self.episode_confirmed = False

    def reset(self):
        self.reset_tracking()
        self.reset_calibration()
        self.yawn_count = 0
        self.last_yawn_time = None

    def _result(
        self,
        yawn_detected=False,
        mouth_ratio=None,
        valid=False,
        yawn_duration=None,
        candidate_duration=None,
        confirmed_duration=None,
        state_transition=None,
        now=None,
    ):
        now = time.monotonic() if now is None else float(now)
        if candidate_duration is None:
            if self.opening_started is None:
                candidate_duration = 0.0
            elif self.confirmed_started is not None:
                candidate_duration = max(
                    0.0,
                    self.confirmed_started - self.opening_started,
                )
            else:
                candidate_duration = max(0.0, now - self.opening_started)
        if confirmed_duration is None:
            confirmed_duration = (
                max(0.0, now - self.confirmed_started)
                if self.confirmed_started is not None
                else 0.0
            )
        return {
            "state": self.state,
            "state_transition": state_transition,
            "yawn_detected": yawn_detected,
            "count": self.yawn_count,
            "mouth_ratio": mouth_ratio,
            "normal_baseline": self.normal_baseline,
            "open_threshold": self.open_threshold,
            "release_threshold": self.release_threshold,
            "yawn_duration": float(
                self.yawn_duration if yawn_duration is None else yawn_duration
            ),
            "candidate_duration": float(candidate_duration),
            "confirmed_duration": float(confirmed_duration),
            "tracking_valid": valid,
        }

    def _log_yawn(self, result):
        state = {
            "MOUTH NORMAL": "NORMAL",
            "MOUTH OPENING": "CANDIDATE",
            "YAWN CANDIDATE": "CANDIDATE",
            "YAWN CONFIRMED": "CONFIRMED",
            "MOUTH RECOVERY": "RECOVERY",
        }.get(result["state"], result["state"])
        print(
            "YAWN: "
            f"state={state} "
            f"mouth_ratio={result['mouth_ratio']} "
            f"candidate_duration={result['candidate_duration']:.3f} "
            f"confirmed_duration={result['confirmed_duration']:.3f} "
            f"event_count={self.yawn_count}"
        )

    def update(self, landmarks, now=None):
        now = time.monotonic() if now is None else float(now)
        ratio = mouth_aspect_ratio(landmarks, MOUTH_INDICES)
        if ratio is None or not np.isfinite(ratio):
            previous_state = self.state
            if (
                not self.episode_confirmed
                and self.last_valid_time is not None
                and now - self.last_valid_time > config.EYE_UNCERTAIN_GRACE_TIME
            ):
                self.reset_tracking()
            self.state = "MOUTH UNCERTAIN"
            result = self._result(
                state_transition=(
                    f"{previous_state} -> {self.state}"
                    if previous_state != self.state
                    else None
                ),
                now=now,
            )
            if config.DEBUG_MODE and result["state_transition"]:
                self._log_yawn(result)
            return result

        gap = (
            None
            if self.last_valid_time is None
            else max(0.0, now - self.last_valid_time)
        )
        if (
            gap is not None
            and gap > config.EYE_UNCERTAIN_GRACE_TIME
            and not self.episode_confirmed
        ):
            self.reset_tracking()
        self.last_valid_time = now

        open_threshold = self.open_threshold
        release_threshold = self.release_threshold

        if self.episode_confirmed:
            previous_state = self.state
            if self.event_counted:
                if ratio <= release_threshold:
                    if self.rearm_started is None:
                        self.rearm_started = now
                    cooldown_elapsed = (
                        self.last_yawn_time is not None
                        and now - self.last_yawn_time >= config.YAWN_COOLDOWN
                    )
                    rearm_elapsed = (
                        now - self.rearm_started >= config.YAWN_REARM_TIME
                    )
                    if cooldown_elapsed and rearm_elapsed:
                        duration = (
                            max(0.0, now - self.opening_started)
                            if self.opening_started is not None
                            else self.yawn_duration
                        )
                        self.reset_tracking()
                        self.last_valid_time = now
                        self.state = "MOUTH NORMAL"
                        result = self._result(
                            mouth_ratio=float(ratio),
                            valid=True,
                            yawn_duration=duration,
                            state_transition=f"{previous_state} -> MOUTH NORMAL",
                            now=now,
                        )
                        if config.DEBUG_MODE:
                            self._log_yawn(result)
                        return result
                else:
                    self.rearm_started = None
                self.state = "MOUTH RECOVERY"
                result = self._result(
                    mouth_ratio=float(ratio),
                    valid=True,
                    state_transition=(
                        f"{previous_state} -> {self.state}"
                        if previous_state != self.state
                        else None
                    ),
                    now=now,
                )
                if config.DEBUG_MODE and result["state_transition"]:
                    self._log_yawn(result)
                return result

            if self.state == "MOUTH UNCERTAIN":
                self.state = (
                    "MOUTH RECOVERY"
                    if self.recovery_started is not None
                    else "YAWN CONFIRMED"
                )
            self.yawn_duration = (
                max(0.0, now - self.opening_started)
                if self.opening_started is not None
                else self.yawn_duration
            )
            if ratio <= release_threshold:
                if self.recovery_started is None:
                    self.recovery_started = now
                self.state = "MOUTH RECOVERY"
                recovery_duration = max(0.0, now - self.recovery_started)
                if recovery_duration >= config.YAWN_RECOVERY_CONFIRM_TIME:
                    self.yawn_count += 1
                    self.last_yawn_time = now
                    self.event_counted = True
                    self.rearm_started = now
                    result = self._result(
                        yawn_detected=True,
                        mouth_ratio=float(ratio),
                        valid=True,
                        state_transition=(
                            f"{previous_state} -> MOUTH RECOVERY"
                            if previous_state != "MOUTH RECOVERY"
                            else None
                        ),
                        now=now,
                    )
                    if config.DEBUG_MODE:
                        self._log_yawn(result)
                    return result
            else:
                self.recovery_started = None
                self.state = "YAWN CONFIRMED"

            result = self._result(
                mouth_ratio=float(ratio),
                valid=True,
                state_transition=(
                    f"{previous_state} -> {self.state}"
                    if previous_state != self.state
                    else None
                ),
                now=now,
            )
            if config.DEBUG_MODE and result["state_transition"]:
                self._log_yawn(result)
            return result

        if self.opening_started is None:
            previous_state = self.state
            if ratio >= open_threshold:
                self.opening_started = now
                self.yawn_duration = 0.0
                self.event_counted = False
                self.confirmed_started = None
                self.recovery_started = None
                self.state = "MOUTH OPENING"
            else:
                self.state = "MOUTH NORMAL"
            result = self._result(
                mouth_ratio=float(ratio),
                valid=True,
                state_transition=(
                    f"{previous_state} -> {self.state}"
                    if previous_state != self.state
                    else None
                ),
                now=now,
            )
            if config.DEBUG_MODE and result["state_transition"]:
                self._log_yawn(result)
            return result

        previous_state = self.state
        self.yawn_duration = max(0.0, now - self.opening_started)
        if ratio >= release_threshold:
            if self.yawn_duration >= config.YAWN_CONFIRM_TIME:
                self.state = "YAWN CONFIRMED"
                self.episode_confirmed = True
                self.confirmed_started = now
                result = self._result(
                    mouth_ratio=float(ratio),
                    valid=True,
                    state_transition=(
                        f"{previous_state} -> {self.state}"
                        if previous_state != self.state
                        else None
                    ),
                    now=now,
                )
                if config.DEBUG_MODE and result["state_transition"]:
                    self._log_yawn(result)
                return result
            self.state = "YAWN CANDIDATE"
            result = self._result(
                mouth_ratio=float(ratio),
                valid=True,
                state_transition=(
                    f"{previous_state} -> {self.state}"
                    if previous_state != self.state
                    else None
                ),
                now=now,
            )
            if config.DEBUG_MODE and result["state_transition"]:
                self._log_yawn(result)
            return result

        duration = self.yawn_duration
        self.state = "MOUTH RECOVERY"
        result = self._result(
            mouth_ratio=float(ratio),
            valid=True,
            yawn_duration=duration,
            state_transition=f"{previous_state} -> MOUTH RECOVERY",
            now=now,
        )
        self.reset_tracking()
        result["state"] = "MOUTH RECOVERY"
        if config.DEBUG_MODE:
            self._log_yawn(result)
        return result

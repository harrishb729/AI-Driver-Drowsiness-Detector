import csv
import itertools
import logging
import os
from datetime import datetime

import config


class SessionLogger:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or config.SESSION_LOG_DIR
        self.file_path = None
        self._disabled = False
        self._event_count = 0
        self._logger = logging.getLogger("driver_drowsiness")
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            self._create_session_file()
        except OSError:
            self._disabled = True
            self._logger.exception(
                "Session logging is unavailable; monitoring will continue."
            )

    def _create_session_file(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"session_{stamp}"
        for suffix in itertools.count():
            name = f"{base_name}.csv" if suffix == 0 else f"{base_name}_{suffix}.csv"
            path = os.path.join(self.log_dir, name)
            try:
                with open(path, mode="x", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "timestamp",
                            "event",
                            "score",
                            "eye_state",
                            "mouth_state",
                            "head_state",
                        ]
                    )
                self.file_path = path
                return
            except FileExistsError:
                continue

    def log_event(self, event, score, eye_state, mouth_state, head_state):
        if self._disabled or self.file_path is None:
            return False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.file_path, mode="a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [timestamp, event, score, eye_state, mouth_state, head_state]
                )
        except OSError:
            self._disabled = True
            self._logger.exception(
                "Session logging failed; monitoring will continue without CSV events."
            )
            return False
        self._event_count += 1
        return True

    @property
    def path(self):
        return self.file_path

    @property
    def event_count(self):
        return self._event_count

    def close(self):
        return None

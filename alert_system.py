import time
import winsound

import config


class AlertSystem:
    def __init__(self):
        self.active = False
        self.last_beep_time = 0.0

    def start_alarm(self):
        if self.active:
            return
        self.active = True
        self.last_beep_time = time.monotonic()
        self._play_alert(200)

    def update(self, now):
        if not self.active:
            return
        if now - self.last_beep_time >= config.ALARM_REPEAT_SECONDS:
            self._play_alert(1200)
            self.last_beep_time = now

    def stop_alarm(self):
        self.active = False
        self.last_beep_time = 0.0

    def is_alarm_active(self):
        return self.active

    def _play_alert(self, frequency):
        try:
            winsound.Beep(frequency, 180)
        except Exception:
            pass

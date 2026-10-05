import cv2
import numpy as np


class Dashboard:
    WIDTH = 1280
    HEIGHT = 720

    BACKGROUND = (18, 23, 30)
    PANEL = (26, 33, 42)
    CARD = (34, 43, 54)
    BORDER = (55, 67, 82)
    TEXT = (235, 241, 247)
    MUTED = (148, 162, 178)
    GREEN = (91, 211, 146)
    AMBER = (45, 177, 255)
    RED = (86, 88, 242)
    BLUE = (227, 174, 94)

    @staticmethod
    def _text(image, text, x, y, scale=0.55, color=None, thickness=1):
        cv2.putText(
            image,
            str(text),
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color or Dashboard.TEXT,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _card(image, rect, fill=None, border=None, radius=12):
        x, y, width, height = (int(value) for value in rect)
        fill = fill or Dashboard.CARD
        border = border or Dashboard.BORDER
        cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), fill, -1)
        cv2.rectangle(image, (x, y + radius), (x + width, y + height - radius), fill, -1)
        for center in (
            (x + radius, y + radius),
            (x + width - radius, y + radius),
            (x + radius, y + height - radius),
            (x + width - radius, y + height - radius),
        ):
            cv2.circle(image, center, radius, fill, -1, cv2.LINE_AA)
        cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), border, 1)
        cv2.rectangle(image, (x, y + radius), (x + width, y + height - radius), border, 1)
        for center in (
            (x + radius, y + radius),
            (x + width - radius, y + radius),
            (x + radius, y + height - radius),
            (x + width - radius, y + height - radius),
        ):
            cv2.circle(image, center, radius, border, 1, cv2.LINE_AA)

    @staticmethod
    def _fit_image(image, target_width, target_height):
        source_height, source_width = image.shape[:2]
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, int(source_width * scale))
        resized_height = max(1, int(source_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
        return canvas

    def render(self, camera_frame, data, debug_mode=False, output_size=None):
        output_width, output_height = output_size or (self.WIDTH, self.HEIGHT)
        output_width = max(900, int(output_width))
        output_height = max(560, int(output_height))
        canvas = np.full((output_height, output_width, 3), self.BACKGROUND, dtype=np.uint8)

        margin = max(18, int(output_width * 0.018))
        header_height = max(76, int(output_height * 0.12))
        footer_height = max(34, int(output_height * 0.055))
        gap = max(14, int(output_width * 0.014))
        content_top = header_height + 12
        content_bottom = output_height - footer_height - 12
        content_height = content_bottom - content_top
        content_width = output_width - margin * 2

        left_width = int((content_width - gap) * 0.65)
        right_width = content_width - gap - left_width
        left_x = margin
        right_x = left_x + left_width + gap

        cv2.rectangle(canvas, (0, 0), (output_width, header_height), (22, 29, 37), -1)
        cv2.line(canvas, (margin, header_height), (output_width - margin, header_height), self.BORDER, 1, cv2.LINE_AA)
        title = "AI DRIVER DROWSINESS DETECTOR"
        subtitle = "REAL-TIME COMPUTER VISION FOR DRIVER SAFETY"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0]
        subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        self._text(canvas, title, (output_width - title_size[0]) // 2, 34, 0.82, self.TEXT, 2)
        self._text(canvas, subtitle, (output_width - subtitle_size[0]) // 2, 59, 0.45, self.MUTED, 1)

        panel_rect = (left_x, content_top, left_width, content_height)
        self._card(canvas, panel_rect, fill=self.PANEL, radius=14)
        self._text(canvas, "LIVE CAMERA", left_x + 20, content_top + 30, 0.62, self.TEXT, 2)
        camera_left = left_x + 16
        camera_top = content_top + 44
        camera_width = left_width - 32
        camera_height = content_height - 60
        camera_view = self._fit_image(camera_frame, camera_width, camera_height)
        canvas[camera_top:camera_top + camera_height, camera_left:camera_left + camera_width] = camera_view
        cv2.rectangle(
            canvas,
            (camera_left, camera_top),
            (camera_left + camera_width - 1, camera_top + camera_height - 1),
            self.BORDER,
            1,
        )

        status = str(data.get("driver_status", "NORMAL")).upper()
        score = max(0, min(100, int(round(float(data.get("drowsiness_score", 0))))))
        eye_state = str(data.get("eye_state", "EYES UNCERTAIN")).upper().replace("EYES ", "")
        blink_count = data.get("blink_count", 0)
        yawn_count = data.get("yawn_count", 0)
        head_state = str(data.get("head_state", "NORMAL")).upper()
        alarm_state = str(data.get("alarm_state", "OFF")).upper()
        session_elapsed = str(data.get("session_elapsed", "00:00:00"))
        event_count = data.get("event_count", 0)
        calibration_state = data.get("calibration_state", "READY")

        status_color = {
            "NORMAL": self.GREEN,
            "CAUTION": self.AMBER,
            "RISK": (0, 198, 255),
            "WARNING": self.RED,
        }.get(status, self.MUTED)
        dashboard_rect = (right_x, content_top, right_width, content_height)
        self._card(canvas, dashboard_rect, fill=self.PANEL, radius=14)
        pad = 16
        card_x = right_x + pad
        card_width = right_width - pad * 2
        self._text(canvas, "DRIVER MONITORING DASHBOARD", card_x + 2, content_top + 27, 0.48, self.TEXT, 1)
        y = content_top + 39
        if debug_mode:
            debug_lines = data.get("debug_lines", ())
            debug_rows = [
                "  |  ".join(str(line) for line in debug_lines[index:index + 2])
                for index in range(0, min(len(debug_lines), 6), 2)
            ]
            for line in debug_rows:
                self._text(canvas, line[:72], card_x + 2, y + 11, 0.29, self.BLUE, 1)
                y += 13
            if debug_rows:
                y += 4

        status_height = max(76, int(content_height * 0.16))
        self._card(canvas, (card_x, y, card_width, status_height))
        self._text(canvas, "DRIVER STATUS", card_x + 15, y + 23, 0.42, self.MUTED, 1)
        self._text(canvas, status, card_x + 15, y + status_height - 15, 0.88, status_color, 2)
        y += status_height + 10

        score_height = max(94, int(content_height * 0.19))
        self._card(canvas, (card_x, y, card_width, score_height))
        self._text(canvas, "DROWSINESS SCORE", card_x + 15, y + 22, 0.42, self.MUTED, 1)
        self._text(canvas, f"{score:02d}", card_x + 15, y + 65, 1.25, self.TEXT, 2)
        self._text(canvas, "/ 100", card_x + 78, y + 64, 0.52, self.MUTED, 1)
        bar_x = card_x + 15
        bar_y = y + score_height - 20
        bar_width = card_width - 30
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + 7), (57, 68, 82), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_width * score / 100), bar_y + 7), status_color, -1)
        y += score_height + 10

        remaining = content_bottom - 12 - y
        gap_y = 8
        tile_height = max(58, (remaining - gap_y * 2) // 3)
        tile_width = (card_width - 8) // 2
        tiles = [
            ("EYES", eye_state, card_x, y),
            ("BLINKS", blink_count, card_x + tile_width + 8, y),
            ("YAWNS", yawn_count, card_x, y + tile_height + gap_y),
            ("HEAD", head_state, card_x + tile_width + 8, y + tile_height + gap_y),
        ]
        for label, value, tile_x, tile_y in tiles:
            self._card(canvas, (tile_x, tile_y, tile_width, tile_height), radius=10)
            self._text(canvas, label, tile_x + 12, tile_y + 21, 0.39, self.MUTED, 1)
            self._text(canvas, str(value), tile_x + 12, tile_y + tile_height - 13, 0.58, self.TEXT, 2)

        bottom_y = y + (tile_height + gap_y) * 2
        bottom_height = max(50, content_bottom - 12 - bottom_y)
        self._card(canvas, (card_x, bottom_y, card_width, bottom_height), radius=10)
        if bottom_height >= 72:
            self._text(canvas, "ALARM", card_x + 12, bottom_y + 20, 0.38, self.MUTED, 1)
            alarm_color = self.RED if alarm_state == "ACTIVE" else self.GREEN
            self._text(canvas, alarm_state, card_x + 12, bottom_y + 42, 0.53, alarm_color, 2)
            column_x = card_x + int(card_width * 0.47)
            self._text(canvas, "SESSION", column_x, bottom_y + 20, 0.38, self.MUTED, 1)
            self._text(canvas, session_elapsed, column_x, bottom_y + 42, 0.5, self.TEXT, 2)
            self._text(canvas, f"EVENTS  {event_count}", column_x, bottom_y + 62, 0.4, self.MUTED, 1)
        else:
            alarm_color = self.RED if alarm_state == "ACTIVE" else self.GREEN
            self._text(canvas, f"ALARM  {alarm_state}", card_x + 12, bottom_y + 23, 0.45, alarm_color, 2)
            self._text(canvas, f"SESSION  {session_elapsed}  |  EVENTS {event_count}", card_x + 12, bottom_y + 43, 0.38, self.TEXT, 1)

        footer_y = output_height - footer_height
        cv2.rectangle(canvas, (0, footer_y), (output_width, output_height), (22, 29, 37), -1)
        cv2.line(canvas, (margin, footer_y), (output_width - margin, footer_y), self.BORDER, 1, cv2.LINE_AA)
        footer_text = f"SYSTEM ONLINE     CAMERA LIVE     AI MODEL ONLINE     AI MONITORING ACTIVE     CALIBRATION: {calibration_state}"
        self._text(canvas, footer_text, margin + 2, footer_y + footer_height // 2 + 5, 0.4, self.GREEN, 1)

        return canvas

import math
from dataclasses import dataclass, field

import dearpygui.dearpygui as dpg

METER_MIN_DB = -48.0
METER_MAX_DB = 0.0
METER_BAR_WIDTH = 8
METER_BAR_GAP = 5
METER_LEFT = 8
METER_WIDTH = 66
METER_SCALE = (0, -6, -12, -24, -48)
METER_GREEN = (65, 210, 90, 255)
METER_YELLOW = (235, 195, 45, 255)
METER_RED = (225, 65, 65, 255)
METER_RANGES = (
    (METER_MIN_DB, -24.0, METER_GREEN),
    (-24.0, -6.0, METER_YELLOW),
    (-6.0, METER_MAX_DB, METER_RED),
)


@dataclass
class InputLevelMeter:
    canvas: int | str
    height_source: int | str
    height_offset: int
    left_db: float = METER_MIN_DB
    right_db: float = METER_MIN_DB
    last_size: tuple[int, int] | None = None
    drawing_items: list[int | str] = field(default_factory=list)

    def set_levels(self, left: float, right: float) -> None:
        """Set channel levels from linear amplitudes in the 0..1 range."""
        self.set_db_levels(self._to_db(left), self._to_db(right))

    def set_db_levels(self, left_db: float, right_db: float) -> None:
        """Set channel levels in dBFS."""
        self.left_db = self._clamp_db(left_db)
        self.right_db = self._clamp_db(right_db)
        self.resize(force=True)

    @staticmethod
    def _to_db(level: float) -> float:
        if level <= 0:
            return METER_MIN_DB
        return 20.0 * math.log10(level)

    @staticmethod
    def _clamp_db(level_db: float) -> float:
        return max(METER_MIN_DB, min(METER_MAX_DB, level_db))

    @staticmethod
    def _level_y(level_db: float, top: float, bottom: float) -> float:
        ratio = (level_db - METER_MIN_DB) / (METER_MAX_DB - METER_MIN_DB)
        return bottom - ratio * (bottom - top)

    def resize(self, *, force: bool = False) -> None:
        size = dpg.get_item_state(self.height_source).get("rect_size")
        if not size:
            return
        width = METER_WIDTH - 1
        height = int(size[1]) + self.height_offset
        if width <= 0 or height <= 0:
            return
        current_size = (width, height)
        if not force and current_size == self.last_size:
            return

        dpg.configure_item(self.canvas, width=width, height=height)
        self.last_size = current_size
        self._redraw(width, height)

    def _redraw(self, width: float, height: float) -> None:
        top = 10
        bottom = max(top + 20, height - 22)
        left_x = METER_LEFT
        right_x = left_x + METER_BAR_WIDTH + METER_BAR_GAP
        scale_x = right_x + METER_BAR_WIDTH + 8

        for item in self.drawing_items:
            if dpg.does_item_exist(item):
                dpg.delete_item(item)
        self.drawing_items.clear()

        for bar_x in (left_x, right_x):
            self.drawing_items.append(
                dpg.draw_rectangle(
                    (bar_x, top),
                    (bar_x + METER_BAR_WIDTH, bottom),
                    color=(75, 75, 75, 255),
                    fill=(25, 25, 25, 255),
                    parent=self.canvas,
                )
            )

        for bar_x, level_db in (
            (left_x, self.left_db),
            (right_x, self.right_db),
        ):
            for range_min, range_max, color in METER_RANGES:
                visible_max = min(level_db, range_max)
                if visible_max <= range_min:
                    continue

                segment_top = self._level_y(visible_max, top, bottom)
                segment_bottom = self._level_y(range_min, top, bottom)
                self.drawing_items.append(
                    dpg.draw_rectangle(
                        (bar_x, segment_top),
                        (bar_x + METER_BAR_WIDTH, segment_bottom),
                        color=color,
                        fill=color,
                        parent=self.canvas,
                    )
                )

        self.drawing_items.extend(
            (
                dpg.draw_text(
                    (left_x + 4, bottom + 4),
                    "L",
                    size=12,
                    parent=self.canvas,
                ),
                dpg.draw_text(
                    (right_x + 3, bottom + 4),
                    "R",
                    size=12,
                    parent=self.canvas,
                ),
            )
        )

        for level_db in METER_SCALE:
            y = self._level_y(level_db, top, bottom)
            self.drawing_items.append(
                dpg.draw_line(
                    (scale_x, y),
                    (scale_x + 5, y),
                    color=(180, 180, 180, 255),
                    parent=self.canvas,
                )
            )
            self.drawing_items.append(
                dpg.draw_text(
                    (scale_x + 8, y - 6),
                    str(abs(level_db)),
                    size=11,
                    parent=self.canvas,
                )
            )

        self.drawing_items.append(
            dpg.draw_text(
                (scale_x + 8, bottom + 4),
                "dB",
                size=11,
                parent=self.canvas,
            )
        )


def add_input_level_meter(
    parent: int | str,
    height_source: int | str,
    height_offset: int,
) -> InputLevelMeter:
    with dpg.drawlist(
        width=-1,
        height=-1,
        parent=parent,
    ) as canvas:
        pass

    return InputLevelMeter(
        canvas=canvas,
        height_source=height_source,
        height_offset=height_offset,
    )

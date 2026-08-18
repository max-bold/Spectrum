import math
from dataclasses import dataclass, field

import dearpygui.dearpygui as dpg

METER_MIN_DB = -48.0
METER_MAX_DB = 0.0
METER_BAR_WIDTH = 8
METER_BAR_GAP = 5
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
class LevelMeter:
    """Compact vertical dBFS meter reusable by measurement module views."""

    canvas: int | str
    height_source: int | str
    labels: tuple[str, ...]
    height_offset: int = 0
    levels_db: list[float] = field(init=False)
    last_size: tuple[int, int] | None = None
    drawing_items: list[int | str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("A level meter needs at least one channel")
        self.levels_db = [METER_MIN_DB] * len(self.labels)

    @property
    def width(self) -> int:
        return 42 + (len(self.labels) - 1) * (
            METER_BAR_WIDTH + METER_BAR_GAP
        )

    def set_levels(self, *levels: float) -> None:
        """Set linear peak amplitudes for all displayed channels."""
        if len(levels) != len(self.labels):
            raise ValueError(
                f"Expected {len(self.labels)} level values, got {len(levels)}"
            )
        self.set_db_levels(*(self._to_db(level) for level in levels))

    def set_db_levels(self, *levels_db: float) -> None:
        """Set dBFS peak levels for all displayed channels."""
        if len(levels_db) != len(self.labels):
            raise ValueError(
                f"Expected {len(self.labels)} level values, got {len(levels_db)}"
            )
        self.levels_db = [self._clamp_db(level) for level in levels_db]
        self.resize(force=True)

    def resize(self, *, force: bool = False) -> None:
        state = dpg.get_item_state(self.height_source)
        size = state.get("rect_size")
        if not size:
            return
        width = self.width
        height = int(size[1]) + self.height_offset
        if height <= 0:
            return
        current_size = (width, height)
        if not force and current_size == self.last_size:
            return

        dpg.configure_item(self.canvas, width=width, height=height)
        self.last_size = current_size
        self._redraw(width, height)

    @staticmethod
    def _to_db(level: float) -> float:
        if level <= 0.0:
            return METER_MIN_DB
        return 20.0 * math.log10(level)

    @staticmethod
    def _clamp_db(level_db: float) -> float:
        return max(METER_MIN_DB, min(METER_MAX_DB, float(level_db)))

    @staticmethod
    def _level_y(level_db: float, top: float, bottom: float) -> float:
        ratio = (level_db - METER_MIN_DB) / (METER_MAX_DB - METER_MIN_DB)
        return bottom - ratio * (bottom - top)

    def _redraw(self, width: float, height: float) -> None:
        top = 10.0
        bottom = max(top + 20.0, height - 22.0)
        scale_x = (
            (len(self.labels) - 1) * (METER_BAR_WIDTH + METER_BAR_GAP)
            + METER_BAR_WIDTH
            + 8
        )

        for item in self.drawing_items:
            if dpg.does_item_exist(item):
                dpg.delete_item(item)
        self.drawing_items.clear()

        for index, (label, level_db) in enumerate(
            zip(self.labels, self.levels_db, strict=True)
        ):
            bar_x = index * (METER_BAR_WIDTH + METER_BAR_GAP)
            self.drawing_items.append(
                dpg.draw_rectangle(
                    (bar_x, top),
                    (bar_x + METER_BAR_WIDTH, bottom),
                    color=(75, 75, 75, 255),
                    fill=(25, 25, 25, 255),
                    parent=self.canvas,
                )
            )
            for range_min, range_max, color in METER_RANGES:
                visible_max = min(level_db, range_max)
                if visible_max <= range_min:
                    continue
                self.drawing_items.append(
                    dpg.draw_rectangle(
                        (bar_x, self._level_y(visible_max, top, bottom)),
                        (bar_x + METER_BAR_WIDTH, self._level_y(range_min, top, bottom)),
                        color=color,
                        fill=color,
                        parent=self.canvas,
                    )
                )
            self.drawing_items.append(
                dpg.draw_text(
                    (bar_x + 1, bottom + 4),
                    label,
                    size=11,
                    parent=self.canvas,
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


def add_level_meter(
    parent: int | str,
    tag: int | str,
    height_source: int | str,
    *,
    labels: tuple[str, ...] = ("1",),
    height_offset: int = 0,
) -> LevelMeter:
    meter = LevelMeter(
        canvas=tag,
        height_source=height_source,
        labels=labels,
        height_offset=height_offset,
    )
    with dpg.drawlist(  # pyright: ignore[reportGeneralTypeIssues]
        width=meter.width,
        height=-1,
        parent=parent,
        tag=tag,
    ):
        pass
    return meter

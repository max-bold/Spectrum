from dataclasses import dataclass, field

import dearpygui.dearpygui as dpg

SPICE_SECTION_COUNT = 10
SPICE_ROW_NAMES = ("L", "C", "R")
SPICE_TABLE_HEIGHT = 100
SPICE_HEADER_HEIGHT = 26
SPICE_ROW_HEIGHT = 24
SPICE_ROW_LABEL_WIDTH = 24
SPICE_TEXT_SIZE = 11
SPICE_HEADERS = (
    "",
    "L1",
    *(f"F{i}" for i in range(1, SPICE_SECTION_COUNT + 1)),
    "R1",
)
GRID_COLOR = (100, 100, 100, 255)
TEXT_COLOR = (220, 220, 220, 255)


@dataclass
class SpiceModelTable:
    canvas: int | str
    width_source: int | str
    l1_value: str = ""
    section_values: tuple[tuple[str, str, str], ...] = field(
        default_factory=lambda: tuple(
            ("", "", "") for _ in range(SPICE_SECTION_COUNT)
        )
    )
    r1_value: str = ""
    last_width: int | None = None
    drawing_items: list[int | str] = field(default_factory=list)

    def set_values(
        self,
        l1: str,
        sections: tuple[tuple[str, str, str], ...],
        r1: str,
    ) -> None:
        if len(sections) != SPICE_SECTION_COUNT:
            raise ValueError(
                f"Expected {SPICE_SECTION_COUNT} sections, got {len(sections)}"
            )
        self.l1_value = l1
        self.section_values = sections
        self.r1_value = r1
        self.resize(force=True)

    def resize(self, *, force: bool = False) -> None:
        size = dpg.get_item_state(self.width_source).get("rect_size")
        if not size:
            return
        width = int(size[0])
        if width <= 0:
            return
        if not force and width == self.last_width:
            return

        dpg.configure_item(
            self.canvas,
            width=width,
            height=SPICE_TABLE_HEIGHT,
        )
        self.last_width = width
        self._redraw(width)

    def _redraw(self, width: float) -> None:
        for item in self.drawing_items:
            if dpg.does_item_exist(item):
                dpg.delete_item(item)
        self.drawing_items.clear()

        right = width - 1
        bottom = SPICE_HEADER_HEIGHT + len(SPICE_ROW_NAMES) * SPICE_ROW_HEIGHT
        data_width = right - SPICE_ROW_LABEL_WIDTH
        column_width = data_width / (len(SPICE_HEADERS) - 1)
        column_edges = [0.0, float(SPICE_ROW_LABEL_WIDTH)]
        column_edges.extend(
            SPICE_ROW_LABEL_WIDTH + column_width * index
            for index in range(1, len(SPICE_HEADERS))
        )

        self._line((0, 0), (right, 0))
        self._line((0, bottom), (right, bottom))
        self._line((0, 0), (0, bottom))
        self._line((right, 0), (right, bottom))

        for x in column_edges[1:]:
            self._line((x, 0), (x, bottom))

        self._line(
            (0, SPICE_HEADER_HEIGHT),
            (right, SPICE_HEADER_HEIGHT),
        )

        l1_left = column_edges[1]
        l1_right = column_edges[2]
        r1_left = column_edges[-2]
        for row_index in range(1, len(SPICE_ROW_NAMES)):
            y = SPICE_HEADER_HEIGHT + row_index * SPICE_ROW_HEIGHT
            self._line((0, y), (l1_left, y))
            self._line((l1_right, y), (r1_left, y))

        for index, header in enumerate(SPICE_HEADERS):
            left = column_edges[index]
            right_edge = column_edges[index + 1]
            self._text_centered(
                header,
                (left + right_edge) / 2,
                SPICE_HEADER_HEIGHT / 2,
                tag=f"spice_header_{header or 'rows'}",
            )

        for index, row_name in enumerate(SPICE_ROW_NAMES):
            center_y = (
                SPICE_HEADER_HEIGHT
                + (index + 0.5) * SPICE_ROW_HEIGHT
            )
            self._text_centered(
                row_name,
                SPICE_ROW_LABEL_WIDTH / 2,
                center_y,
                tag=f"spice_row_{row_name}",
            )

        body_center_y = (
            SPICE_HEADER_HEIGHT
            + len(SPICE_ROW_NAMES) * SPICE_ROW_HEIGHT / 2
        )
        self._draw_value(
            self.l1_value,
            (column_edges[1] + column_edges[2]) / 2,
            body_center_y,
            tag="spice_value_L1",
        )
        for section_index, section_values in enumerate(
            self.section_values,
            start=1,
        ):
            center_x = (
                column_edges[section_index + 1]
                + column_edges[section_index + 2]
            ) / 2
            for row_index, value in enumerate(section_values):
                center_y = (
                    SPICE_HEADER_HEIGHT
                    + (row_index + 0.5) * SPICE_ROW_HEIGHT
                )
                self._draw_value(
                    value,
                    center_x,
                    center_y,
                    tag=f"spice_value_F{section_index}_{SPICE_ROW_NAMES[row_index]}",
                )
        self._draw_value(
            self.r1_value,
            (column_edges[-2] + column_edges[-1]) / 2,
            body_center_y,
            tag="spice_value_R1",
        )

    def _line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> None:
        self.drawing_items.append(
            dpg.draw_line(
                start,
                end,
                color=GRID_COLOR,
                parent=self.canvas,
            )
        )

    def _text_centered(
        self,
        text: str,
        center_x: float,
        center_y: float,
        *,
        tag: str,
    ) -> None:
        text_width, text_height = dpg.get_text_size(text)
        self.drawing_items.append(
            dpg.draw_text(
                (
                    center_x - text_width / 2,
                    center_y - text_height / 2,
                ),
                text,
                tag=tag,
                color=TEXT_COLOR,
                size=SPICE_TEXT_SIZE,
                parent=self.canvas,
            )
        )

    def _draw_value(
        self,
        value: str,
        center_x: float,
        center_y: float,
        *,
        tag: str,
    ) -> None:
        if value:
            self._text_centered(
                value,
                center_x,
                center_y,
                tag=tag,
            )


def add_spice_model_table(
    parent: int | str,
    width_source: int | str,
) -> SpiceModelTable:
    with dpg.drawlist(
        width=-1,
        height=SPICE_TABLE_HEIGHT,
        parent=parent,
    ) as canvas:
        pass

    return SpiceModelTable(
        canvas=canvas,
        width_source=width_source,
    )

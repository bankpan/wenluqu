from __future__ import annotations

from html.parser import HTMLParser
from typing import List


class HTMLTableParser(HTMLParser):
    """Small HTML table parser that expands row/col spans into a dense grid."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self._current_row: List[str] | None = None
        self._current_col: int = 0
        self._in_cell = False
        self._cell_text: List[str] = []
        self._cell_rowspan = 1
        self._cell_colspan = 1
        self._row_span_map: dict[int, dict[str, object]] = {}

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._start_row()
        elif tag in {"td", "th"}:
            self._start_cell(dict(attrs))
        elif tag == "br" and self._in_cell:
            self._cell_text.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in {"td", "th"} and self._in_cell:
            self._end_cell()
        elif tag == "tr":
            self._end_row()

    def handle_data(self, data: str):
        if self._in_cell:
            self._cell_text.append(data)

    def _start_row(self) -> None:
        if self._current_row is not None:
            # Implicitly close an open row to avoid malformed html issues.
            self._end_row()
        self._current_row = []
        self._current_col = 0
        self._consume_active_spans()

    def _end_row(self) -> None:
        if self._current_row is None:
            return
        # Fill any remaining active spans at the end of row.
        self._consume_active_spans()
        # Strip trailing empty values for cleanliness while preserving alignment.
        while self._current_row and (self._current_row[-1] or "").strip() == "":
            self._current_row.pop()
        self.rows.append(self._current_row)
        self._current_row = None
        self._current_col = 0

    def _start_cell(self, attrs: dict[str, str]) -> None:
        self._in_cell = True
        self._cell_text = []
        self._cell_rowspan = int(attrs.get("rowspan") or 1)
        self._cell_colspan = int(attrs.get("colspan") or 1)

    def _end_cell(self) -> None:
        text = "".join(self._cell_text).strip()
        self._place_cell(text, self._cell_rowspan, self._cell_colspan)
        self._in_cell = False
        self._cell_text = []
        self._cell_rowspan = 1
        self._cell_colspan = 1

    def _place_cell(self, text: str, rowspan: int, colspan: int) -> None:
        self._advance_to_free_column()
        for offset in range(colspan):
            value = text if offset == 0 else ""
            if self._current_row is None:
                self._current_row = []
            self._current_row.append(value)
            if rowspan > 1:
                self._row_span_map[self._current_col] = {
                    "value": value,
                    "rows_left": rowspan - 1,
                }
            self._current_col += 1

    def _consume_active_spans(self) -> None:
        while self._row_span_map.get(self._current_col):
            span = self._row_span_map[self._current_col]
            if self._current_row is None:
                self._current_row = []
            self._current_row.append(span["value"])
            span["rows_left"] -= 1
            if span["rows_left"] == 0:
                del self._row_span_map[self._current_col]
            self._current_col += 1

    def _advance_to_free_column(self) -> None:
        while self._row_span_map.get(self._current_col):
            self._consume_active_spans()


def parse_table_html(html: str) -> List[List[str]]:
    parser = HTMLTableParser()
    parser.feed(html)
    if parser._current_row is not None:
        parser._end_row()
    return [row for row in parser.rows if any((cell or "").strip() for cell in row)]

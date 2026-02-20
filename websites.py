from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


HEADER_RE = re.compile(r"^#\s+([^\s]+)\s+(\d+)\s*$")
COMMENT_RE = re.compile(r"^<!--.*-->$")


@dataclass(frozen=True)
class WebsiteQuery:
    site: str
    number: int
    query: str
    disabled: bool = False

    @property
    def section_id(self) -> str:
        return f"{self.site} {self.number}"

    @property
    def dir_name(self) -> str:
        return f"{sanitize_name(self.site)}_{self.number}"

    @property
    def session_dir_name(self) -> str:
        return f"{sanitize_name(self.site)}_session"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "site"


def parse_websites_md(path: Path) -> list[WebsiteQuery]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        if current.get("query") is None:
            raise ValueError(f"Section '{current['site']} {current['number']}' is missing a query line")
        sections.append(current)
        current = None

    for i, raw in enumerate(raw_lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if COMMENT_RE.match(line):
            continue

        header_match = HEADER_RE.match(line)
        if header_match:
            flush_current()
            current = {
                "site": header_match.group(1),
                "number": int(header_match.group(2)),
                "disabled": False,
                "query": None,
                "line": i,
            }
            continue

        if current is None:
            raise ValueError(f"Line {i}: content found outside section header: {line}")

        if line == "- disabled":
            current["disabled"] = True
            continue

        if current.get("query") is None:
            current["query"] = line
            continue

        raise ValueError(
            f"Line {i}: section '{current['site']} {current['number']}' has multiple query lines"
        )

    flush_current()

    out = [
        WebsiteQuery(
            site=str(s["site"]),
            number=int(s["number"]),
            query=str(s["query"]),
            disabled=bool(s["disabled"]),
        )
        for s in sections
    ]

    # Validate site+number uniqueness.
    seen: set[tuple[str, int]] = set()
    for item in out:
        key = (item.site, item.number)
        if key in seen:
            raise ValueError(f"Duplicate section header: {item.site} {item.number}")
        seen.add(key)

    return out

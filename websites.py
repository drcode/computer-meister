from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HEADER_RE = re.compile(r"^#\s+([^\s]+)\s+(\d+)\s*$")
COMMENT_RE = re.compile(r"^<!--.*-->$")
DEFAULT_BROWSER = "firefox"
CHROMIUM_BROWSER = "chromium"
SUPPORTED_BROWSERS = {DEFAULT_BROWSER, CHROMIUM_BROWSER}


@dataclass(frozen=True)
class WebsiteQuery:
    site: str
    number: int
    query: str
    disabled: bool = False
    browser: str = DEFAULT_BROWSER
    nofill: bool = False
    nocapture: bool = False

    @property
    def section_id(self) -> str:
        return f"{self.site} {self.number}"

    @property
    def dir_name(self) -> str:
        return f"{sanitize_name(self.site)}_{self.number}"

    @property
    def session_dir_name(self) -> str:
        site_name = sanitize_name(self.site)
        if self.browser == DEFAULT_BROWSER:
            return f"{site_name}_session"
        return f"{site_name}_{sanitize_name(self.browser)}_session"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "site"


def _validate_unique_headers(queries: Iterable[WebsiteQuery]) -> None:
    seen: set[tuple[str, int]] = set()
    for item in queries:
        key = (item.site, item.number)
        if key in seen:
            raise ValueError(f"Duplicate section header: {item.site} {item.number}")
        seen.add(key)


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
                "browser": DEFAULT_BROWSER,
                "nofill": False,
                "nocapture": False,
                "query": None,
                "line": i,
            }
            continue

        if current is None:
            raise ValueError(f"Line {i}: content found outside section header: {line}")

        if line == "- disabled":
            current["disabled"] = True
            continue

        if line == "- chromium":
            current["browser"] = CHROMIUM_BROWSER
            continue

        if line == "- nofill":
            current["nofill"] = True
            continue

        if line == "- nocapture":
            current["nocapture"] = True
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
            browser=str(s["browser"]),
            nofill=bool(s["nofill"]),
            nocapture=bool(s["nocapture"]),
        )
        for s in sections
    ]

    _validate_unique_headers(out)
    return out


def render_websites_md(queries: list[WebsiteQuery]) -> str:
    _validate_unique_headers(queries)
    lines: list[str] = []
    for query in queries:
        site = query.site.strip()
        question = query.query.strip()
        if not site:
            raise ValueError("site cannot be empty when writing websites.md")
        if not question:
            raise ValueError(f"Section '{query.section_id}' must have a non-empty query line")
        if query.browser not in SUPPORTED_BROWSERS:
            raise ValueError(f"Unsupported browser '{query.browser}' in section '{query.section_id}'")

        lines.append(f"# {site} {query.number}")
        if query.disabled:
            lines.append("- disabled")
        if query.browser == CHROMIUM_BROWSER:
            lines.append("- chromium")
        if query.nofill:
            lines.append("- nofill")
        if query.nocapture:
            lines.append("- nocapture")
        lines.append(question)
        lines.append("")

    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def write_websites_md(path: Path, queries: list[WebsiteQuery]) -> None:
    text = render_websites_md(queries)
    path.write_text(text, encoding="utf-8")

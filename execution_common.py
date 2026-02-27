from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from planning import PlanCommand, render_plan
from websites import CHROMIUM_BROWSER, DEFAULT_BROWSER, WebsiteQuery


DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 1024
FIREFOX_ARGS: list[str] = []
CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-http2",
    "--disable-quic",
]
COMPUTER_USE_MODEL = "computer-use-preview"
ANSWER_MODEL = "gpt-5.2"
SESSION_STORAGE_FILE = "session_storage.json"
COOKIES_FILE = "cookies.json"
GLOBAL_LOGIN_INFO_FILE = Path.home() / "computer_meister_login_info.json"
FIREFOX_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:147.0) Gecko/20100101 Firefox/147.0.2"
FIREFOX_STEALTH_PREFS = {
    "dom.webdriver.enabled": False,
}


@dataclass(frozen=True)
class QueryOutcome:
    query: WebsiteQuery
    success: bool
    skipped: bool
    results_path: Path
    pertinent_text: str
    elapsed_ms: int
    headless: bool
    help_used: bool


class ManualLoginRequiredError(RuntimeError):
    """Raised when interactive login is required but disabled by caller mode."""


class LockRegistry:
    def __init__(self) -> None:
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()
        self.login_prompt_lock = threading.Lock()

    def session_lock_for(self, session_key: str) -> threading.Lock:
        key = session_key.strip().lower()
        with self._session_locks_guard:
            lock = self._session_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[key] = lock
            return lock


def _artifact_index(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        return int(parts[-1])
    return -1


def _redact_image_data_url(value: str) -> str | None:
    if not value.startswith("data:image/"):
        return None
    if ";base64," not in value:
        return "<redacted image data url>"
    header, payload = value.split(",", 1)
    mime_match = re.match(r"^data:(image/[^;]+);base64$", header, flags=re.IGNORECASE)
    mime = mime_match.group(1) if mime_match else "image/unknown"
    return f"<redacted image data url mime={mime} base64_chars={len(payload)}>"


def _sanitize_for_log(value: Any, *, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        redacted = _redact_image_data_url(value)
        return redacted if redacted is not None else value
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, Path):
        return str(value)

    marker = id(value)
    if marker in _seen:
        return "<cycle>"
    _seen.add(marker)
    try:
        if isinstance(value, dict):
            return {str(key): _sanitize_for_log(val, _seen=_seen) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_sanitize_for_log(item, _seen=_seen) for item in value]

        if hasattr(value, "model_dump"):
            try:
                return _sanitize_for_log(value.model_dump(), _seen=_seen)
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return _sanitize_for_log(value.__dict__, _seen=_seen)
            except Exception:
                pass
        return str(value)
    finally:
        _seen.discard(marker)


def _dump_json(value: Any) -> str:
    return json.dumps(_sanitize_for_log(value), ensure_ascii=False, indent=2)


def _launch_context_kwargs(*, browser: str, headless: bool) -> dict[str, Any]:
    if browser == CHROMIUM_BROWSER:
        args = list(CHROMIUM_ARGS)
        base = {
            "channel": CHROMIUM_BROWSER,
            "args": args,
        }

        if not headless:
            return {
                **base,
                "headless": False,
                "no_viewport": True,
                "args": args + [f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"],
            }

        return {
            **base,
            "headless": True,
            "viewport": {"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
            "device_scale_factor": 1,
        }

    if browser != DEFAULT_BROWSER:
        raise ValueError(f"Unsupported browser '{browser}'")

    args = list(FIREFOX_ARGS)
    base = {"args": args, "user_agent": FIREFOX_USER_AGENT}

    if not headless:
        return {
            **base,
            "headless": False,
            "no_viewport": True,
            "firefox_user_prefs": FIREFOX_STEALTH_PREFS,
            "args": args + [f"--window-size={DISPLAY_WIDTH},{DISPLAY_HEIGHT}"],
        }

    return {
        **base,
        "headless": True,
        "viewport": {"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
        "device_scale_factor": 1,
    }


def _responses_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            maybe_text = getattr(content, "text", None)
            if isinstance(maybe_text, str) and maybe_text.strip():
                parts.append(maybe_text.strip())
    return "\n".join(parts).strip()


def _screenshot_is_single_color(png_bytes: bytes) -> bool:
    if not png_bytes:
        return True
    try:
        from io import BytesIO

        from PIL import Image  # type: ignore[import-not-found]

        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        width, height = img.size
        if width <= 0 or height <= 0:
            return True
        base = img.getpixel((0, 0))

        # Sample a grid (plus implicit corners/center) to avoid false negatives on small spinners.
        sample_x = 50
        sample_y = 50
        denom_x = max(1, sample_x - 1)
        denom_y = max(1, sample_y - 1)
        for iy in range(sample_y):
            y = (iy * (height - 1)) // denom_y
            for ix in range(sample_x):
                x = (ix * (width - 1)) // denom_x
                if img.getpixel((x, y)) != base:
                    return False
        return True
    except Exception:
        # Solid-color PNGs compress extremely well; use a size heuristic as a last resort.
        return len(png_bytes) < 15000


async def _safe_goto(page: Page, url: str, *, timeout_ms: int = 35000) -> None:
    candidates = [url]
    if not url.startswith(("http://", "https://")):
        candidates.insert(0, f"https://{url}")

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            await page.goto(candidate, wait_until="domcontentloaded", timeout=timeout_ms)
            return
        except PlaywrightTimeoutError as exc:
            last_error = exc

            async def _accept_if_rendered() -> bool:
                try:
                    await page.wait_for_timeout(1000)
                except Exception:
                    pass
                try:
                    png_bytes = await page.screenshot()
                except Exception:
                    return False
                if _screenshot_is_single_color(png_bytes):
                    return False
                return True

            # If the page rendered something meaningful, proceed even if DOMContentLoaded never fired.
            if await _accept_if_rendered():
                return

            # If we only got a blank/solid-color frame, retry with a weaker wait condition.
            try:
                await page.goto(candidate, wait_until="commit", timeout=timeout_ms)
                if await _accept_if_rendered():
                    return
            except Exception as commit_exc:  # noqa: BLE001
                last_error = commit_exc
                if await _accept_if_rendered():
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error:
        raise last_error
    raise RuntimeError(f"could not navigate to {url}")


def _normalize_key(raw_key: Any) -> str:
    key = str(raw_key).strip()
    if not key:
        return key

    alias = {
        "enter": "Enter",
        "return": "Enter",
        "esc": "Escape",
        "escape": "Escape",
        "tab": "Tab",
        "space": "Space",
        "spacebar": "Space",
        "backspace": "Backspace",
        "delete": "Delete",
        "del": "Delete",
        "insert": "Insert",
        "home": "Home",
        "end": "End",
        "pageup": "PageUp",
        "pagedown": "PageDown",
        "up": "ArrowUp",
        "down": "ArrowDown",
        "left": "ArrowLeft",
        "right": "ArrowRight",
        "arrowup": "ArrowUp",
        "arrowdown": "ArrowDown",
        "arrowleft": "ArrowLeft",
        "arrowright": "ArrowRight",
        "shift": "Shift",
        "ctrl": "Control",
        "control": "Control",
        "alt": "Alt",
        "option": "Alt",
        "meta": "Meta",
        "cmd": "Meta",
        "command": "Meta",
    }

    def normalize_piece(piece: str) -> str:
        p = piece.strip()
        if not p:
            return p
        return alias.get(p.lower(), p if len(p) == 1 else p[:1].upper() + p[1:].lower())

    if "+" in key:
        return "+".join(normalize_piece(piece) for piece in key.split("+"))
    return normalize_piece(key)


def _plan_command_line(command: PlanCommand) -> str:
    return render_plan([command]).strip()


def _plan_command_payload(command: PlanCommand) -> dict[str, Any]:
    return {
        "name": command.name,
        "args": list(command.args),
        "line": _plan_command_line(command),
    }

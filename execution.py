from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from playwright.async_api import BrowserContext, Page, async_playwright

from ai import load_openai_api_key, openai
from html_preprocessor import preprocess_html
from planning import PlanCommand
from websites import WebsiteQuery


DISPLAY_WIDTH = 2048
DISPLAY_HEIGHT = 1600
CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-http2",
    "--disable-quic",
]
COMPUTER_USE_MODEL = "computer-use-preview"
ANSWER_MODEL = "gpt-5.2"


@dataclass(frozen=True)
class QueryOutcome:
    query: WebsiteQuery
    success: bool
    results_path: Path
    pertinent_text: str
    elapsed_ms: int
    headless: bool
    help_used: bool


class LockRegistry:
    def __init__(self) -> None:
        self._site_locks: dict[str, threading.Lock] = {}
        self._site_locks_guard = threading.Lock()
        self.login_prompt_lock = threading.Lock()

    def site_lock_for(self, site: str) -> threading.Lock:
        key = site.lower().strip()
        with self._site_locks_guard:
            lock = self._site_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._site_locks[key] = lock
            return lock


class ArtifactRecorder:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.counter = 0
        self.events: list[dict[str, Any]] = []

    async def capture(self, page: Page, *, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        idx = self.counter
        self.counter += 1

        screenshot_name = f"screenshot_{idx}.png"
        html_name = f"page_{idx}.html"
        screenshot_path = self.artifact_dir / screenshot_name
        html_path = self.artifact_dir / html_name

        screenshot_bytes = await page.screenshot(path=str(screenshot_path))
        html = await page.content()
        html_path.write_text(html, encoding="utf-8")

        event = {
            "index": idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "screenshot": screenshot_name,
            "page_html": html_name,
        }
        if metadata:
            event["metadata"] = metadata

        self.events.append(event)
        return {
            "event": event,
            "screenshot_bytes": screenshot_bytes,
            "screenshot_path": screenshot_path,
            "html_path": html_path,
        }

    def latest_screenshots(self, count: int = 4) -> list[Path]:
        pairs = sorted(self.artifact_dir.glob("screenshot_*.png"), key=_artifact_index)
        return pairs[-count:]

    def latest_pages(self, count: int = 4) -> list[Path]:
        pairs = sorted(self.artifact_dir.glob("page_*.html"), key=_artifact_index)
        return pairs[-count:]


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


def _launch_context_kwargs(headless: bool) -> dict[str, Any]:
    return {
        "headless": headless,
        "viewport": {"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
        "device_scale_factor": 1,
        "args": CHROMIUM_ARGS,
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


async def _safe_goto(page: Page, url: str, *, timeout_ms: int = 35000) -> None:
    candidates = [url]
    if not url.startswith(("http://", "https://")):
        candidates.insert(0, f"https://{url}")

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            await page.goto(candidate, wait_until="domcontentloaded", timeout=timeout_ms)
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


async def _execute_computer_action(page: Page, action: dict[str, Any], *, allow_text_entry: bool) -> None:
    action_type = action.get("type")

    if action_type == "click":
        await page.mouse.click(action["x"], action["y"], button=action.get("button", "left"))
    elif action_type == "double_click":
        await page.mouse.dblclick(action["x"], action["y"])
    elif action_type == "type":
        if not allow_text_entry:
            raise RuntimeError("text entry attempted but enable_text_entry was not set")
        await page.keyboard.type(action["text"])
    elif action_type == "keypress":
        for key in action.get("keys", []):
            await page.keyboard.press(_normalize_key(key))
    elif action_type == "scroll":
        await page.mouse.move(action.get("x", DISPLAY_WIDTH // 2), action.get("y", DISPLAY_HEIGHT // 2))
        await page.mouse.wheel(action.get("scroll_x", 0), action.get("scroll_y", 0))
    elif action_type == "drag":
        path = action.get("path") or []
        if len(path) < 2:
            return
        await page.mouse.move(path[0]["x"], path[0]["y"])
        await page.mouse.down()
        for point in path[1:]:
            await page.mouse.move(point["x"], point["y"])
        await page.mouse.up()
    elif action_type == "wait":
        await asyncio.sleep(max(0, int(action.get("ms", 500))) / 1000.0)
    elif action_type == "move":
        await page.mouse.move(action["x"], action["y"])
    elif action_type == "screenshot":
        return
    else:
        raise RuntimeError(f"unsupported computer action: {action_type}")


class PlanExecutor:
    def __init__(
        self,
        *,
        query: WebsiteQuery,
        commands: list[PlanCommand],
        artifact_dir: Path,
        session_dir: Path,
        login_prompt_lock: threading.Lock,
    ) -> None:
        self.query = query
        self.commands = commands
        self.artifact_dir = artifact_dir
        self.session_dir = session_dir
        self.login_prompt_lock = login_prompt_lock

        self.recorder = ArtifactRecorder(artifact_dir)
        self.answer_outputs: list[str] = []
        self.allow_text_entry = False
        self.headless = True
        self.help_used = False
        self.target_url = f"https://{query.site}"
        self._llm_call_counts: dict[str, int] = {}

        self.openai_client = OpenAI(api_key=load_openai_api_key())

        self._playwright = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        self.exploration_runs: list[dict[str, Any]] = []

    def _next_llm_index(self, prefix: str) -> int:
        value = self._llm_call_counts.get(prefix, 0)
        self._llm_call_counts[prefix] = value + 1
        return value

    def _write_llm_call_log(self, prefix: str, request: Any, response: Any, *, index: int | None = None) -> None:
        if index is None:
            index = self._next_llm_index(prefix)
        payload = {
            "llm_call": prefix,
            "index": index,
            "provider": "openai",
            "request": request,
            "response": response,
        }
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / f"{prefix}_{index}.json").write_text(_dump_json(payload), encoding="utf-8")

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("page is not initialized")
        return self._page

    async def _launch_context(self, *, headless: bool) -> None:
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            str(self.session_dir),
            **_launch_context_kwargs(headless=headless),
        )
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

    async def _close_context(self) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None
            self._page = None

    async def _shutdown(self) -> None:
        await self._close_context()
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def run(self) -> str:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        await self._launch_context(headless=True)
        try:
            for command in self.commands:
                await self._execute_command(command)

            if not self.answer_outputs:
                raise RuntimeError("plan finished without any answer command output")
            return "\n\n".join(output.strip() for output in self.answer_outputs if output.strip()).strip()
        finally:
            await self._shutdown()

    async def _execute_command(self, command: PlanCommand) -> None:
        name = command.name

        if name == "target_site":
            self.target_url = str(command.args[0])
            await _safe_goto(self.page, self.target_url)
            await self.recorder.capture(self.page, source="target_site", metadata={"url": self.target_url})
            return

        if name == "login_required":
            await self._ensure_login()
            await self.recorder.capture(self.page, source="login_required")
            return

        if name == "enable_text_entry":
            self.allow_text_entry = True
            return

        if name == "explore_website_openai":
            instruction = str(command.args[0])
            max_steps = int(command.args[1])
            await self._run_exploration(instruction, max_steps)
            return

        if name == "click":
            await self.page.mouse.click(int(command.args[0]), int(command.args[1]), button="left")
            await self.recorder.capture(
                self.page,
                source="click",
                metadata={"x": int(command.args[0]), "y": int(command.args[1])},
            )
            return

        if name == "type":
            if not self.allow_text_entry:
                raise RuntimeError("type command encountered before enable_text_entry")
            text = str(command.args[0])
            await self.page.keyboard.type(text)
            await self.recorder.capture(self.page, source="type", metadata={"text": text})
            return

        if name == "wait":
            ms = int(command.args[0])
            await asyncio.sleep(max(0, ms) / 1000.0)
            await self.recorder.capture(self.page, source="wait", metadata={"ms": ms})
            return

        if name == "vscroll":
            delta = int(command.args[0])
            await self.page.mouse.wheel(0, delta)
            await self.recorder.capture(self.page, source="vscroll", metadata={"delta": delta})
            return

        if name == "answer_query_images":
            answer = await self._answer_query_images(str(command.args[0]))
            self.answer_outputs.append(answer)
            return

        if name == "answer_query_text":
            answer = await self._answer_query_text(str(command.args[0]))
            self.answer_outputs.append(answer)
            return

        raise RuntimeError(f"unsupported plan command: {name}")

    async def _ensure_login(self) -> None:
        await _safe_goto(self.page, self.target_url)
        is_logged_in = await self._is_logged_in_page(self.page, self.target_url)
        if is_logged_in:
            return

        await self._close_context()

        self.login_prompt_lock.acquire()
        try:
            self.headless = False
            self.help_used = True
            await self._launch_context(headless=False)
            await _safe_goto(self.page, self.target_url)
            print(
                f"\nLogin required for {self.query.section_id}. "
                "Please complete login in the opened browser, then press ENTER here to continue.",
                flush=True,
            )
            await asyncio.get_event_loop().run_in_executor(None, input)
        finally:
            self.login_prompt_lock.release()
            await self._close_context()

        await self._launch_context(headless=True)
        await _safe_goto(self.page, self.target_url)

    async def _is_logged_in_page(self, page: Page, url: str) -> bool:
        screenshot = await page.screenshot()
        screenshot_b64 = base64.b64encode(screenshot).decode("ascii")
        prompt = (
            "Is the user already logged in on this webpage? "
            "Look for account/avatar/sign out indicators. "
            "Respond with exactly YES or NO."
        )
        request = {
            "model": ANSWER_MODEL,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"URL: {url}\n{prompt}"},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{screenshot_b64}"},
                    ],
                }
            ],
            "max_output_tokens": 12,
        }

        try:
            response = self.openai_client.responses.create(**request)
            self._write_llm_call_log("is_logged_in_check", request, response)
        except Exception:
            self._write_llm_call_log("is_logged_in_check", request, "error: openai request failed")
            return False

        text = _responses_text(response).strip().lower()
        return text.startswith("yes")

    async def _run_exploration(self, instruction: str, max_steps: int) -> None:
        max_steps = max(1, min(max_steps, 80))
        prompt = (
            "Explore this website to gather information for later answering.\n"
            f"Query target: {self.query.query}\n"
            f"Exploration goal: {instruction}\n"
            "Stop once you have gathered enough context and then return a concise summary.\n"
        )
        if not self.allow_text_entry:
            prompt += "Do not enter text into inputs or search fields.\n"

        initial = await self.recorder.capture(self.page, source="explore_start", metadata={"instruction": instruction})
        initial_b64 = base64.b64encode(initial["screenshot_bytes"]).decode("ascii")

        tools = [
            {
                "type": "computer_use_preview",
                "display_width": DISPLAY_WIDTH,
                "display_height": DISPLAY_HEIGHT,
                "environment": "browser",
            }
        ]

        exploration_loop_index = self._next_llm_index("exploration_loop")
        initial_request = {
            "model": COMPUTER_USE_MODEL,
            "tools": tools,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{initial_b64}"},
                    ],
                }
            ],
            "truncation": "auto",
        }
        response = self.openai_client.responses.create(**initial_request)
        self._write_llm_call_log("exploration_loop", initial_request, response, index=exploration_loop_index)

        steps_out: list[dict[str, Any]] = []
        final_text = ""

        for step_idx in range(max_steps):
            computer_call = None
            text_parts: list[str] = []
            for item in getattr(response, "output", []) or []:
                item_type = getattr(item, "type", None)
                if item_type == "computer_call":
                    computer_call = item
                elif item_type == "text":
                    maybe_text = getattr(item, "text", "")
                    if isinstance(maybe_text, str) and maybe_text.strip():
                        text_parts.append(maybe_text.strip())

            if computer_call is None:
                final_text = "\n".join(text_parts).strip() or getattr(response, "output_text", "") or ""
                break

            action_obj = computer_call.action
            action = action_obj.model_dump() if hasattr(action_obj, "model_dump") else dict(action_obj)
            action_error = None
            try:
                await _execute_computer_action(self.page, action, allow_text_entry=self.allow_text_entry)
            except Exception as exc:  # noqa: BLE001
                action_error = str(exc)

            captured = await self.recorder.capture(
                self.page,
                source="explore_action",
                metadata={"step": step_idx, "action": action, "error": action_error},
            )
            screenshot_b64 = base64.b64encode(captured["screenshot_bytes"]).decode("ascii")

            step_entry = {
                "step": step_idx,
                "action": action,
                "error": action_error,
                "artifact": captured["event"],
            }
            steps_out.append(step_entry)

            call_output: dict[str, Any] = {
                "type": "computer_call_output",
                "call_id": computer_call.call_id,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                },
            }

            safety_checks = getattr(computer_call, "pending_safety_checks", []) or []
            if safety_checks:
                call_output["acknowledged_safety_checks"] = [
                    {"id": check.id, "code": check.code, "message": check.message}
                    for check in safety_checks
                ]

            followup_request = {
                "model": COMPUTER_USE_MODEL,
                "tools": tools,
                "previous_response_id": response.id,
                "input": [call_output],
                "truncation": "auto",
            }
            try:
                response = self.openai_client.responses.create(**followup_request)
                self._write_llm_call_log(
                    "exploration_loop",
                    followup_request,
                    response,
                    index=self._next_llm_index("exploration_loop"),
                )
            except Exception as exc:
                self._write_llm_call_log(
                    "exploration_loop",
                    followup_request,
                    f"error: {exc}",
                    index=self._next_llm_index("exploration_loop"),
                )
                raise

        run_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instruction": instruction,
            "max_command_steps": max_steps,
            "allow_text_entry": self.allow_text_entry,
            "final_text": final_text,
            "steps": steps_out,
        }
        self.exploration_runs.append(run_record)
        exploration_path = self.artifact_dir / "exploration_steps.json"
        exploration_path.write_text(
            json.dumps({"runs": self.exploration_runs}, indent=2),
            encoding="utf-8",
        )

    async def _answer_query_text(self, instruction: str) -> str:
        pages = self.recorder.latest_pages(4)
        if not pages:
            capture = await self.recorder.capture(self.page, source="answer_query_text_autocapture")
            pages = [capture["html_path"]]

        page_chunks: list[str] = []
        for page_path in pages:
            raw = page_path.read_text(encoding="utf-8", errors="replace")
            try:
                processed = preprocess_html(raw)
            except Exception:
                processed = raw
            processed = processed[:45000]
            page_chunks.append(f"FILE: {page_path.name}\n{processed}")

        prompt = (
            "Answer the query strictly from the provided webpage HTML snapshots.\n"
            f"Original query: {self.query.query}\n"
            f"Answer instruction: {instruction}\n\n"
            "If the evidence is insufficient or contradictory, start with FAIL and explain what is missing.\n\n"
            + "\n\n".join(page_chunks)
        )
        request = {
            "model": ANSWER_MODEL,
            "system": "Provide a concise factual answer grounded in the supplied snapshots.",
            "max_output_tokens": 1200,
            "temperature": 0,
            "prompt": prompt,
        }

        try:
            response = openai(
                prompt,
                system="Provide a concise factual answer grounded in the supplied snapshots.",
                max_output_tokens=1200,
                temperature=0,
            )
            self._write_llm_call_log("answer_query_text", request, response)
        except Exception:
            self._write_llm_call_log("answer_query_text", request, "error: openai request failed")
            raise
        return str(response).strip()

    async def _answer_query_images(self, instruction: str) -> str:
        screenshots = self.recorder.latest_screenshots(4)
        if not screenshots:
            capture = await self.recorder.capture(self.page, source="answer_query_images_autocapture")
            screenshots = [capture["screenshot_path"]]

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Answer the query from the provided webpage screenshots.\n"
                    f"Original query: {self.query.query}\n"
                    f"Answer instruction: {instruction}\n"
                    "If evidence is insufficient, start with FAIL and explain what is missing."
                ),
            }
        ]

        for image_path in screenshots:
            b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                }
            )

        request = {
            "model": ANSWER_MODEL,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 1200,
        }
        try:
            response = self.openai_client.responses.create(**request)
            self._write_llm_call_log("answer_query_images", request, response)
        except Exception:
            self._write_llm_call_log("answer_query_images", request, "error: openai request failed")
            raise
        text = _responses_text(response).strip()
        if text:
            return text
        return "FAIL\nNo text response was returned from image analysis."


def write_results_file(
    *,
    path: Path,
    elapsed_ms: int,
    headless: bool,
    help_used: bool,
    body: str,
) -> None:
    text = (
        f"- time: {elapsed_ms}\n"
        f"- headless: {headless}\n"
        f"- help: {help_used}\n"
        "---\n"
        f"{body.strip()}\n"
    )
    path.write_text(text, encoding="utf-8")


def body_from_results_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "\n---\n"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def run_query_execution(
    *,
    query: WebsiteQuery,
    commands: list[PlanCommand],
    artifact_dir: Path,
    results_path: Path,
    session_dir: Path,
    login_prompt_lock: threading.Lock,
) -> QueryOutcome:
    start = time.perf_counter()
    success = False
    headless = True
    help_used = False
    body: str

    try:
        executor = PlanExecutor(
            query=query,
            commands=commands,
            artifact_dir=artifact_dir,
            session_dir=session_dir,
            login_prompt_lock=login_prompt_lock,
        )
        body = asyncio.run(executor.run())
        headless = executor.headless
        help_used = executor.help_used
        success = not body.lstrip().startswith("FAIL")
        if not body.strip():
            body = "FAIL\nNo answer text was produced by the plan."
            success = False
    except Exception as exc:  # noqa: BLE001
        err = "".join(traceback.format_exception(exc)).strip()
        body = f"FAIL\n{err}"
        success = False

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    write_results_file(
        path=results_path,
        elapsed_ms=elapsed_ms,
        headless=headless,
        help_used=help_used,
        body=body,
    )

    pertinent = body_from_results_file(results_path)
    return QueryOutcome(
        query=query,
        success=success,
        results_path=results_path,
        pertinent_text=pertinent,
        elapsed_ms=elapsed_ms,
        headless=headless,
        help_used=help_used,
    )

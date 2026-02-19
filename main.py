#!/usr/bin/env python3
"""Computer Meister — a CLI that uses Playwright + OpenAI computer-use-preview to answer questions about websites."""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from anthropic import Anthropic
from openai import OpenAI
from playwright.async_api import async_playwright

# ── Constants ────────────────────────────────────────────────────────────────

DISPLAY_WIDTH = 2048
DISPLAY_HEIGHT = 1600
MAX_ACTIONS = 40
TIMEOUT_SECONDS = 400
PROFILE_BASE_DIR = Path.home() / ".computer-meister-profile"
ARTIFACTS_DIR = Path("artifacts")
CUA_MODEL = "computer-use-preview"
CLAUDE_CUA_MODEL = "claude-sonnet-4-5"
CLAUDE_COMPUTER_BETA = "computer-use-2025-01-24"
LOGIN_CHECK_MODEL = "gpt-4o-mini"

DEFAULT_URL = "https://weather.com"
DEFAULT_QUESTION = "What is the current temperature for ZIP code 94941?"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 Safari/537.36"
)
CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-http2",
    "--disable-quic",
]


# ── API key loading (mirrors ~/monorepo/common/ai.py) ───────────────────────

def _find_key_file(filename: str, *, extra_paths: list[Path] | None = None) -> str | None:
    candidates: list[Path] = []
    if extra_paths:
        candidates.extend(extra_paths)
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in [base, *list(base.parents)]:
            candidates.append(parent / filename)
    for path in candidates:
        try:
            if path.exists():
                key = path.read_text(encoding="utf-8").strip()
                if key:
                    return key
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=1)
def load_openai_api_key() -> str:
    v = os.environ.get("OPENAI_API_KEY")
    if v and v.strip():
        return v.strip()
    extra_paths = [
        Path("/opt/openai_api_key.txt"),
        Path.home() / ".openai_api_key.txt",
    ]
    key = _find_key_file("openai_api_key.txt", extra_paths=extra_paths)
    if key:
        return key
    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY or place openai_api_key.txt in/above the working directory."
    )


@functools.lru_cache(maxsize=1)
def load_anthropic_api_key() -> str:
    v = os.environ.get("ANTHROPIC_API_KEY")
    if v and v.strip():
        return v.strip()
    extra_paths = [
        Path("/opt/anthropic_api_key.txt"),
        Path.home() / ".anthropic_api_key.txt",
    ]
    key = _find_key_file("anthropic_api_key.txt", extra_paths=extra_paths)
    if key:
        return key
    raise RuntimeError(
        "Anthropic API key not found. Set ANTHROPIC_API_KEY or place anthropic_api_key.txt in/above the working directory."
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def add_www_subdomain(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host or host.startswith("www."):
        return url
    netloc = parsed.netloc
    if parsed.port:
        netloc = f"www.{host}:{parsed.port}"
    else:
        netloc = f"www.{host}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def navigation_candidates(url: str) -> list[str]:
    base = ensure_url(url)
    parsed = urlparse(base)
    scheme = parsed.scheme or "https"
    host = parsed.hostname or ""
    path = parsed.path or "/"
    query = parsed.query
    fragment = parsed.fragment
    port = f":{parsed.port}" if parsed.port else ""

    hosts: list[str] = []
    if host:
        hosts.append(host)
        if not host.startswith("www."):
            hosts.append(f"www.{host}")

    schemes = [scheme]

    candidates: list[str] = []
    for candidate_scheme in schemes:
        for candidate_host in hosts:
            candidate = urlunparse(
                (candidate_scheme, f"{candidate_host}{port}", path, parsed.params, query, fragment)
            )
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


async def goto_with_fallback(page, url: str, *, wait_until: str = "domcontentloaded", timeout: int = 30000):
    last_error: Exception | None = None
    candidates = navigation_candidates(url)
    for idx, candidate in enumerate(candidates):
        try:
            if idx == 0:
                return await page.goto(candidate, wait_until=wait_until, timeout=timeout)
            log(f"retrying navigation with {candidate}")
            return await page.goto(candidate, wait_until=wait_until, timeout=timeout)
        except Exception as e:
            last_error = e
            log(f"goto failed for {candidate}: {e}")

    if wait_until != "commit":
        for candidate in candidates:
            try:
                log(f"retrying navigation with relaxed wait: {candidate}")
                return await page.goto(candidate, wait_until="commit", timeout=timeout)
            except Exception as e:
                last_error = e
                log(f"goto failed (relaxed) for {candidate}: {e}")

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Navigation failed with no candidates for URL: {url}")


def get_profile_key(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return "default"
    if host in {"localhost", "127.0.0.1"}:
        return host
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def get_profile_paths(url: str) -> tuple[Path, Path]:
    profile_key = get_profile_key(url)
    profile_dir = PROFILE_BASE_DIR / profile_key
    return profile_dir, profile_dir / "storage_state.json"


async def take_screenshot(page, artifacts_dir: Path, step: int | str = "") -> tuple[str, bytes]:
    """Take a screenshot and return (base64_str, raw_bytes)."""
    raw = await page.screenshot()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = artifacts_dir / f"screenshot_{ts}_{step}.png"
    path.write_bytes(raw)
    return base64.b64encode(raw).decode(), raw


def get_artifact_dir(url: str) -> Path:
    return ARTIFACTS_DIR / get_profile_key(url)


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def launch_context_kwargs(*, headless: bool) -> dict:
    kwargs = {
        "headless": headless,
        "user_agent": DEFAULT_USER_AGENT,
        "args": CHROMIUM_ARGS,
        "viewport": {"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
        "device_scale_factor": 1,
    }
    return kwargs


# ── Action execution ────────────────────────────────────────────────────────

async def execute_action(page, action: dict) -> None:
    action_type = action["type"]

    if action_type == "click":
        x, y = action["x"], action["y"]
        button = action.get("button", "left")
        log(f"click ({x}, {y}) button={button}")
        await page.mouse.click(x, y, button=button)

    elif action_type == "double_click":
        x, y = action["x"], action["y"]
        log(f"double_click ({x}, {y})")
        await page.mouse.dblclick(x, y)

    elif action_type == "type":
        text = action["text"]
        display = text if len(text) <= 40 else text[:37] + "..."
        log(f'type "{display}"')
        await page.keyboard.type(text)

    elif action_type == "keypress":
        keys = action["keys"]
        log(f"keypress {keys}")
        for key in keys:
            await page.keyboard.press(key)

    elif action_type == "scroll":
        x, y = action["x"], action["y"]
        dx, dy = action["scroll_x"], action["scroll_y"]
        log(f"scroll at ({x}, {y}) delta=({dx}, {dy})")
        await page.mouse.move(x, y)
        await page.mouse.wheel(dx, dy)

    elif action_type == "drag":
        path = action["path"]
        log(f"drag {path}")
        start = path[0]
        await page.mouse.move(start["x"], start["y"])
        await page.mouse.down()
        for pt in path[1:]:
            await page.mouse.move(pt["x"], pt["y"])
        await page.mouse.up()

    elif action_type == "screenshot":
        log("screenshot (requested by model)")

    elif action_type == "wait":
        ms = action.get("ms", 500)
        log(f"wait {ms}ms")
        await asyncio.sleep(ms / 1000)

    elif action_type == "move":
        x, y = action["x"], action["y"]
        log(f"move ({x}, {y})")
        await page.mouse.move(x, y)

    else:
        log(f"unknown action type: {action_type}")


# ── Claude computer-use helpers ─────────────────────────────────────────────

def _coord_from_input(payload: dict[str, Any], key: str = "coordinate") -> tuple[int, int] | None:
    coord = payload.get(key)
    if isinstance(coord, list) and len(coord) >= 2:
        return int(coord[0]), int(coord[1])
    return None


def claude_action_to_local_action(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload.get("action")

    if action in {"left_click", "right_click", "middle_click", "double_click"}:
        xy = _coord_from_input(payload)
        if not xy:
            raise ValueError(f"Missing coordinate for {action}")
        x, y = xy
        if action == "double_click":
            return {"type": "double_click", "x": x, "y": y}
        button = "left" if action == "left_click" else "right" if action == "right_click" else "middle"
        return {"type": "click", "x": x, "y": y, "button": button}

    if action == "left_click_drag":
        start = _coord_from_input(payload, "start_coordinate")
        end = _coord_from_input(payload, "end_coordinate")
        if not start or not end:
            raise ValueError("Missing drag coordinates")
        return {
            "type": "drag",
            "path": [
                {"x": start[0], "y": start[1]},
                {"x": end[0], "y": end[1]},
            ],
        }

    if action == "mouse_move":
        xy = _coord_from_input(payload)
        if not xy:
            raise ValueError("Missing coordinate for mouse_move")
        return {"type": "move", "x": xy[0], "y": xy[1]}

    if action == "key":
        keys = payload.get("keys")
        if isinstance(keys, list) and keys:
            return {"type": "keypress", "keys": [str(k) for k in keys]}
        text = payload.get("text")
        if isinstance(text, str) and text:
            return {"type": "keypress", "keys": [text]}
        raise ValueError("Missing keys/text for key action")

    if action == "type":
        text = payload.get("text")
        if not isinstance(text, str):
            raise ValueError("Missing text for type action")
        return {"type": "type", "text": text}

    if action == "scroll":
        xy = _coord_from_input(payload) or (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2)
        amount = int(payload.get("scroll_amount", 600))
        # Claude may emit small line-based amounts (e.g. 2/3). Playwright wheel
        # expects pixel-ish deltas, so normalize tiny values to useful movement.
        if amount == 0:
            amount = 600
        elif abs(amount) <= 10:
            amount *= 120
        direction = str(payload.get("scroll_direction", "down")).lower()
        dx, dy = 0, amount
        if direction == "up":
            dy = -amount
        elif direction == "left":
            dx, dy = -amount, 0
        elif direction == "right":
            dx, dy = amount, 0
        return {"type": "scroll", "x": xy[0], "y": xy[1], "scroll_x": dx, "scroll_y": dy}

    if action == "screenshot":
        return {"type": "screenshot"}

    if action == "wait":
        return {"type": "wait", "ms": int(payload.get("duration_ms", 500))}

    raise ValueError(f"Unsupported Claude action: {action}")


def claude_tool_result_with_screenshot(tool_use_id: str, screenshot_b64: str, error: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if error:
        content.append({"type": "text", "text": error})
    content.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": screenshot_b64,
            },
        }
    )
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


# ── Login check ──────────────────────────────────────────────────────────────

def check_logged_in(client: OpenAI, screenshot_b64: str, url: str) -> bool:
    """Ask a vision model whether the page looks like the user is logged in."""
    resp = client.chat.completions.create(
        model=LOGIN_CHECK_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"I'm looking at {url}. Does this page show that a user is logged in? "
                            "Look for indicators like a username, avatar, account menu, 'My Account', 'Sign Out', etc. "
                            "Reply with ONLY 'yes' or 'no'."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                    },
                ],
            }
        ],
        max_tokens=10,
    )
    answer = resp.choices[0].message.content.strip().lower()
    return answer.startswith("yes")


# ── Human login flow ────────────────────────────────────────────────────────

async def human_login_flow(
    playwright,
    client: OpenAI | None,
    url: str,
    profile_dir: Path,
    storage_state_path: Path,
    artifacts_dir: Path,
) -> None:
    """Check if logged in; if not, open a visible browser for manual login."""
    log("Checking login status...")
    if client is None:
        log("No OpenAI key available for visual login check; opening manual login directly.")
        print("\n>>> Please log in in the browser, then come back here and press ENTER to continue. <<<\n")
        ctx = await playwright.chromium.launch_persistent_context(
            str(profile_dir),
            **launch_context_kwargs(headless=False),
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        try:
            await goto_with_fallback(page, url, wait_until="commit", timeout=30000)
        except Exception as e:
            log(f"Manual login auto-navigation failed: {e}")
            print(f">>> Please manually navigate to {url} in the opened browser before pressing ENTER. <<<\n")
        await asyncio.get_event_loop().run_in_executor(None, input)
        await ctx.storage_state(path=str(storage_state_path))
        await ctx.close()
        log("Browser closed. Session saved.")
        return

    async def _restore_cookies(context):
        if storage_state_path.exists():
            try:
                state = json.loads(storage_state_path.read_text())
                cookies = state.get("cookies", [])
                if cookies:
                    await context.add_cookies(cookies)
            except Exception:
                pass

    async def _try_login_check_navigation(context, page_obj):
        """Try navigating for login check, escalating from headless to headed."""
        try:
            await goto_with_fallback(page_obj, url, wait_until="commit", timeout=15000)
            return context, page_obj
        except Exception as e:
            log(f"Headless login check failed: {e}")

        # Try headed mode with bundled Chromium
        await context.close()
        log("Retrying login check in headed mode.")
        context = await playwright.chromium.launch_persistent_context(
            str(profile_dir),
            **launch_context_kwargs(headless=False),
        )
        page_obj = context.pages[0] if context.pages else await context.new_page()
        await _restore_cookies(context)
        try:
            await goto_with_fallback(page_obj, url, wait_until="commit", timeout=20000)
            return context, page_obj
        except Exception as e:
            log(f"Headed login check failed: {e}")
        raise

    ctx = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        **launch_context_kwargs(headless=True),
    )
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
    await _restore_cookies(ctx)

    try:
        ctx, page = await _try_login_check_navigation(ctx, page)
    except Exception as e:
        log(f"All login check navigation attempts failed: {e}")
        try:
            await ctx.close()
        except Exception:
            pass

        log("Opening browser for manual login...")
        print("\n>>> Please log in in the browser, then come back here and press ENTER to continue. <<<\n")
        ctx = await playwright.chromium.launch_persistent_context(
            str(profile_dir),
            **launch_context_kwargs(headless=False),
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await _restore_cookies(ctx)
        try:
            await goto_with_fallback(page, url, wait_until="commit", timeout=30000)
        except Exception as e2:
            log(f"Manual login auto-navigation failed: {e2}")
            print(f">>> Please manually navigate to {url} in the opened browser before pressing ENTER. <<<\n")

        await asyncio.get_event_loop().run_in_executor(None, input)
        await ctx.storage_state(path=str(storage_state_path))
        await ctx.close()
        log("Browser closed. Session saved.")
        return
    await asyncio.sleep(2)
    screenshot_b64, _ = await take_screenshot(page, artifacts_dir, "login_check")
    logged_in = check_logged_in(client, screenshot_b64, url)
    await ctx.close()

    if logged_in:
        log("Already logged in — proceeding.")
        return

    log("Not logged in. Opening browser for manual login...")
    print("\n>>> Please log in in the browser, then come back here and press ENTER to continue. <<<\n")
    ctx = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        **launch_context_kwargs(headless=False),
    )
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
    try:
        await goto_with_fallback(page, url, wait_until="commit", timeout=30000)
    except Exception as e:
        log(f"Manual login auto-navigation failed: {e}")
        print(f">>> Please manually navigate to {url} in the opened browser before pressing ENTER. <<<\n")

    # Wait for user to finish logging in — pressing Enter ensures we close cleanly
    await asyncio.get_event_loop().run_in_executor(None, input)

    # Explicitly save cookies + localStorage before closing
    await ctx.storage_state(path=str(storage_state_path))
    await ctx.close()
    log("Browser closed. Session saved.")


# ── CUA loop ────────────────────────────────────────────────────────────────

async def cua_loop(
    playwright,
    client: OpenAI,
    url: str,
    question: str,
    profile_dir: Path,
    storage_state_path: Path,
    artifacts_dir: Path,
    clicks_only: bool = False,
) -> str:
    """Run the Computer Use Agent loop and return the final answer."""
    async def restore_storage_state(context) -> None:
        if not storage_state_path.exists():
            return
        try:
            state = json.loads(storage_state_path.read_text())
            cookies = state.get("cookies", [])
            if cookies:
                await context.add_cookies(cookies)
                log(f"Restored {len(cookies)} saved cookies.")
        except Exception as e:
            log(f"Warning: could not restore saved state: {e}")

    ctx = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        **launch_context_kwargs(headless=True),
    )
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()

    # Restore saved cookies/localStorage if available
    await restore_storage_state(ctx)

    try:
        print(f"\nNavigating to {url} ...")
        try:
            await goto_with_fallback(page, url, wait_until="domcontentloaded", timeout=30000)
        except Exception as nav_error:
            log(f"Headless navigation failed: {nav_error}")
            log("Retrying navigation in headed mode.")
            await ctx.close()
            ctx = await playwright.chromium.launch_persistent_context(
                str(profile_dir),
                **launch_context_kwargs(headless=False),
            )
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            await restore_storage_state(ctx)
            await goto_with_fallback(page, url, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(2)

        screenshot_b64, _ = await take_screenshot(page, artifacts_dir, "initial")

        tools = [
            {
                "type": "computer_use_preview",
                "display_width": DISPLAY_WIDTH,
                "display_height": DISPLAY_HEIGHT,
                "environment": "browser",
            }
        ]

        # Initial request: user message + screenshot
        response = client.responses.create(
            model=CUA_MODEL,
            tools=tools,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    ],
                }
            ],
            truncation="auto",
        )

        start_time = time.time()
        action_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT_SECONDS:
                print(f"\nTimeout reached ({TIMEOUT_SECONDS}s). Stopping.")
                break

            # Find computer_call items in output
            computer_call = None
            text_parts = []
            for item in response.output:
                if getattr(item, "type", None) == "computer_call":
                    computer_call = item
                elif getattr(item, "type", None) == "text":
                    text_parts.append(item.text)

            # No computer call → model is done
            if computer_call is None:
                answer = "\n".join(text_parts) if text_parts else getattr(response, "output_text", "")
                return answer

            action_count += 1
            if action_count > MAX_ACTIONS:
                print(f"\nMax actions ({MAX_ACTIONS}) reached. Stopping.")
                # Extract any text we have so far
                return "\n".join(text_parts) if text_parts else "Max actions reached without a final answer."

            action = computer_call.action
            action_dict = action.model_dump() if hasattr(action, "model_dump") else dict(action)
            call_id = computer_call.call_id

            print(f"\n[Action {action_count}/{MAX_ACTIONS}] ", end="")
            if clicks_only and action_dict.get("type") != "click":
                log(f"filtered non-click action: {action_dict.get('type')}")
            else:
                await execute_action(page, action_dict)

            # Brief pause for page to settle
            await asyncio.sleep(0.5)
            screenshot_b64, _ = await take_screenshot(page, artifacts_dir, f"action_{action_count}")

            # Build the follow-up input
            call_output: dict = {
                "type": "computer_call_output",
                "call_id": call_id,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                },
            }

            # Acknowledge any pending safety checks
            safety_checks = getattr(computer_call, "pending_safety_checks", [])
            if safety_checks:
                call_output["acknowledged_safety_checks"] = [
                    {"id": sc.id, "code": sc.code, "message": sc.message}
                    for sc in safety_checks
                ]

            response = client.responses.create(
                model=CUA_MODEL,
                tools=tools,
                previous_response_id=response.id,
                input=[call_output],
                truncation="auto",
            )

        # Fallback: extract whatever text the model returned
        text_parts = []
        for item in response.output:
            if getattr(item, "type", None) == "text":
                text_parts.append(item.text)
        return "\n".join(text_parts) if text_parts else "No answer obtained."

    finally:
        await ctx.close()


async def cua_loop_claude(
    playwright,
    client: Anthropic,
    url: str,
    question: str,
    profile_dir: Path,
    storage_state_path: Path,
    artifacts_dir: Path,
    clicks_only: bool = False,
) -> str:
    """Run Claude computer-use loop and return the final answer."""
    async def restore_storage_state(context) -> None:
        if not storage_state_path.exists():
            return
        try:
            state = json.loads(storage_state_path.read_text())
            cookies = state.get("cookies", [])
            if cookies:
                await context.add_cookies(cookies)
                log(f"Restored {len(cookies)} saved cookies.")
        except Exception as e:
            log(f"Warning: could not restore saved state: {e}")

    ctx = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        **launch_context_kwargs(headless=True),
    )
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
    await restore_storage_state(ctx)

    try:
        print(f"\nNavigating to {url} ...")
        try:
            await goto_with_fallback(page, url, wait_until="domcontentloaded", timeout=30000)
        except Exception as nav_error:
            log(f"Headless navigation failed: {nav_error}")
            log("Retrying navigation in headed mode.")
            await ctx.close()
            ctx = await playwright.chromium.launch_persistent_context(
                str(profile_dir),
                **launch_context_kwargs(headless=False),
            )
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            await restore_storage_state(ctx)
            await goto_with_fallback(page, url, wait_until="domcontentloaded", timeout=45000)
        await asyncio.sleep(2)

        screenshot_b64, _ = await take_screenshot(page, artifacts_dir, "initial")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                ],
            }
        ]

        tools = [
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": DISPLAY_WIDTH,
                "display_height_px": DISPLAY_HEIGHT,
                "display_number": 1,
            }
        ]

        start_time = time.time()
        action_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > TIMEOUT_SECONDS:
                print(f"\nTimeout reached ({TIMEOUT_SECONDS}s). Stopping.")
                return "Timeout reached before a final answer."

            response = client.beta.messages.create(
                model=CLAUDE_CUA_MODEL,
                max_tokens=1024,
                messages=messages,
                tools=tools,
                betas=[CLAUDE_COMPUTER_BETA],
            )

            tool_uses = [item for item in response.content if getattr(item, "type", None) == "tool_use"]
            text_parts = [item.text for item in response.content if getattr(item, "type", None) == "text"]

            if not tool_uses:
                answer = "\n".join(text_parts).strip()
                return answer or "No answer obtained."

            assistant_content: list[dict[str, Any]] = []
            for item in response.content:
                item_type = getattr(item, "type", None)
                if item_type == "text":
                    assistant_content.append({"type": "text", "text": item.text})
                elif item_type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": item.id,
                            "name": item.name,
                            "input": item.input,
                        }
                    )
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            for tool_use in tool_uses:
                action_count += 1
                if action_count > MAX_ACTIONS:
                    print(f"\nMax actions ({MAX_ACTIONS}) reached. Stopping.")
                    answer = "\n".join(text_parts).strip()
                    return answer or "Max actions reached without a final answer."

                action_payload = dict(tool_use.input)
                action_error: str | None = None

                print(f"\n[Action {action_count}/{MAX_ACTIONS}] ", end="")
                try:
                    local_action = claude_action_to_local_action(action_payload)
                    is_click = local_action.get("type") == "click"
                    if clicks_only and not is_click:
                        log(f"filtered non-click action: {local_action.get('type')}")
                    else:
                        await execute_action(page, local_action)
                except Exception as e:
                    action_error = f"Action execution error: {e}"
                    log(action_error)

                await asyncio.sleep(0.5)
                screenshot_b64, _ = await take_screenshot(page, artifacts_dir, f"action_{action_count}")
                tool_results.append(
                    claude_tool_result_with_screenshot(tool_use.id, screenshot_b64, error=action_error)
                )

            messages.append({"role": "user", "content": tool_results})

    finally:
        await ctx.close()


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_browser_query_async(
    url: str,
    question: str,
    *,
    human_login: bool = False,
    claude: bool = False,
    clicks_only: bool = False,
) -> str:
    url = ensure_url(url)
    artifacts_dir = get_artifact_dir(url)

    profile_dir, storage_state_path = get_profile_paths(url)
    profile_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "url": url,
        "question": question,
        "human_login": human_login,
        "claude": claude,
        "clicks_only": clicks_only,
        "started_at": datetime.now().isoformat(),
        "actions": [],
    }

    async with async_playwright() as pw:
        login_check_client: OpenAI | None = None
        if claude:
            anthropic_client = Anthropic(api_key=load_anthropic_api_key())
            if human_login:
                try:
                    login_check_client = OpenAI(api_key=load_openai_api_key())
                except Exception:
                    login_check_client = None
        else:
            openai_client = OpenAI(api_key=load_openai_api_key())
            login_check_client = openai_client

        if human_login:
            await human_login_flow(
                pw, login_check_client, url, profile_dir, storage_state_path, artifacts_dir
            )

        if claude:
            answer = await cua_loop_claude(
                pw,
                anthropic_client,
                url,
                question,
                profile_dir,
                storage_state_path,
                artifacts_dir,
                clicks_only=clicks_only,
            )
        else:
            answer = await cua_loop(
                pw,
                openai_client,
                url,
                question,
                profile_dir,
                storage_state_path,
                artifacts_dir,
                clicks_only=clicks_only,
            )

    run_meta["finished_at"] = datetime.now().isoformat()
    run_meta["artifacts_dir"] = str(artifacts_dir)
    run_meta["answer"] = answer

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = artifacts_dir / f"run_{ts}.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))
    return answer


def run_browser_query(
    url: str,
    question: str,
    *,
    human_login: bool = False,
    claude: bool = False,
    clicks_only: bool = False,
) -> str:
    return asyncio.run(
        run_browser_query_async(
            url=url,
            question=question,
            human_login=human_login,
            claude=claude,
            clicks_only=clicks_only,
        )
    )


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Computer Meister — browse the web with AI to answer questions."
    )
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="Target URL")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="Question to answer")
    parser.add_argument(
        "--human-login",
        action="store_true",
        help="Check login status and allow manual login if needed",
    )
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Use Anthropic Claude computer-use agent instead of OpenAI computer-use-preview",
    )
    parser.add_argument(
        "--clicks-only",
        action="store_true",
        help="Filter out all non-click model actions",
    )
    args = parser.parse_args()

    url = ensure_url(args.url)
    question = args.question
    human_login = args.human_login
    claude = args.claude
    clicks_only = args.clicks_only

    print(f"URL:      {url}")
    print(f"Question: {question}")
    if human_login:
        print("Mode:     human-login enabled")
    if claude:
        print("Mode:     claude enabled")
    if clicks_only:
        print("Mode:     clicks-only enabled")

    answer = await run_browser_query_async(
        url=url,
        question=question,
        human_login=human_login,
        claude=claude,
        clicks_only=clicks_only,
    )

    print(f"\n{'=' * 60}")
    print("ANSWER:")
    print(answer)
    print(f"{'=' * 60}")
    print(f"\nArtifacts saved to {get_artifact_dir(url)}/")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

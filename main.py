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
from urllib.parse import urlparse

from openai import OpenAI
from playwright.async_api import async_playwright

# ── Constants ────────────────────────────────────────────────────────────────

DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 768
MAX_ACTIONS = 10
TIMEOUT_SECONDS = 120
PROFILE_BASE_DIR = Path.home() / ".computer-meister-profile"
ARTIFACTS_DIR = Path("artifacts")
CUA_MODEL = "computer-use-preview"
LOGIN_CHECK_MODEL = "gpt-4o-mini"

DEFAULT_URL = "https://weather.com"
DEFAULT_QUESTION = "What is the current temperature for ZIP code 94941?"


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


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


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


async def take_screenshot(page, step: int | str = "") -> tuple[str, bytes]:
    """Take a screenshot and return (base64_str, raw_bytes)."""
    raw = await page.screenshot()
    # Save artifact
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ARTIFACTS_DIR / f"screenshot_{ts}_{step}.png"
    path.write_bytes(raw)
    return base64.b64encode(raw).decode(), raw


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


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
        ms = action.get("ms", 1000)
        log(f"wait {ms}ms")
        await asyncio.sleep(ms / 1000)

    elif action_type == "move":
        x, y = action["x"], action["y"]
        log(f"move ({x}, {y})")
        await page.mouse.move(x, y)

    else:
        log(f"unknown action type: {action_type}")


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

async def human_login_flow(playwright, client: OpenAI, url: str, profile_dir: Path, storage_state_path: Path) -> None:
    """Check if logged in; if not, open a visible browser for manual login."""
    log("Checking login status...")
    ctx = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        headless=True,
        viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
    )
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()

    # Restore saved cookies if available
    if storage_state_path.exists():
        try:
            state = json.loads(storage_state_path.read_text())
            cookies = state.get("cookies", [])
            if cookies:
                await ctx.add_cookies(cookies)
        except Exception:
            pass

    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await asyncio.sleep(2)
    screenshot_b64, _ = await take_screenshot(page, "login_check")
    logged_in = check_logged_in(client, screenshot_b64, url)
    await ctx.close()

    if logged_in:
        log("Already logged in — proceeding.")
        return

    log("Not logged in. Opening browser for manual login...")
    print("\n>>> Please log in in the browser, then come back here and press ENTER to continue. <<<\n")
    ctx = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        headless=False,
        viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
    )
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)

    # Wait for user to finish logging in — pressing Enter ensures we close cleanly
    await asyncio.get_event_loop().run_in_executor(None, input)

    # Explicitly save cookies + localStorage before closing
    await ctx.storage_state(path=str(storage_state_path))
    await ctx.close()
    log("Browser closed. Session saved.")


# ── CUA loop ────────────────────────────────────────────────────────────────

async def cua_loop(playwright, client: OpenAI, url: str, question: str, profile_dir: Path, storage_state_path: Path) -> str:
    """Run the Computer Use Agent loop and return the final answer."""
    ctx = await playwright.chromium.launch_persistent_context(
        str(profile_dir),
        headless=True,
        viewport={"width": DISPLAY_WIDTH, "height": DISPLAY_HEIGHT},
    )
    page = ctx.pages[0] if ctx.pages else await ctx.new_page()

    # Restore saved cookies/localStorage if available
    if storage_state_path.exists():
        try:
            state = json.loads(storage_state_path.read_text())
            cookies = state.get("cookies", [])
            if cookies:
                await ctx.add_cookies(cookies)
                log(f"Restored {len(cookies)} saved cookies.")
        except Exception as e:
            log(f"Warning: could not restore saved state: {e}")

    try:
        print(f"\nNavigating to {url} ...")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(2)

        screenshot_b64, _ = await take_screenshot(page, "initial")

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
            await execute_action(page, action_dict)

            # Brief pause for page to settle
            await asyncio.sleep(0.5)
            screenshot_b64, _ = await take_screenshot(page, f"action_{action_count}")

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


# ── Main ─────────────────────────────────────────────────────────────────────

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
    args = parser.parse_args()

    url = ensure_url(args.url)
    question = args.question
    human_login = args.human_login

    print(f"URL:      {url}")
    print(f"Question: {question}")
    if human_login:
        print("Mode:     human-login enabled")

    api_key = load_openai_api_key()
    client = OpenAI(api_key=api_key)

    profile_dir, storage_state_path = get_profile_paths(url)
    profile_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "url": url,
        "question": question,
        "human_login": human_login,
        "started_at": datetime.now().isoformat(),
        "actions": [],
    }

    async with async_playwright() as pw:
        if human_login:
            await human_login_flow(pw, client, url, profile_dir, storage_state_path)

        answer = await cua_loop(pw, client, url, question, profile_dir, storage_state_path)

    run_meta["finished_at"] = datetime.now().isoformat()
    run_meta["answer"] = answer

    # Save run metadata
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = ARTIFACTS_DIR / f"run_{ts}.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))

    print(f"\n{'=' * 60}")
    print("ANSWER:")
    print(answer)
    print(f"{'=' * 60}")
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from openai import OpenAI
from playwright.async_api import BrowserContext, Page, async_playwright

from ai import load_openai_api_key
from execution_common import (
    COOKIES_FILE,
    GLOBAL_LOGIN_INFO_FILE,
    SESSION_STORAGE_FILE,
    CHROMIUM_BROWSER,
    LockRegistry,
    ManualLoginRequiredError,
    QueryOutcome,
    _dump_json,
    _launch_context_kwargs,
    _plan_command_line,
    _plan_command_payload,
    _safe_goto,
)
from execution_content import AnsweringMixin, ArtifactRecorder
from execution_exploration import ExplorationMixin, _execute_plan_interaction
from execution_login import LoginMixin
from planning import PlanCommand
from websites import WebsiteQuery


class PlanExecutor(LoginMixin, ExplorationMixin, AnsweringMixin):
    def __init__(
        self,
        *,
        query: WebsiteQuery,
        commands: list[PlanCommand],
        artifact_dir: Path,
        session_dir: Path,
        login_prompt_lock: threading.Lock,
        allow_manual_login: bool,
    ) -> None:
        self.query = query
        self.commands = commands
        self.artifact_dir = artifact_dir
        self.session_dir = session_dir
        self.login_prompt_lock = login_prompt_lock
        self.allow_manual_login = allow_manual_login

        self.recorder = ArtifactRecorder(artifact_dir)
        self.answer_outputs: list[str] = []
        self.allow_text_entry = False
        self.headless = True
        self.help_used = False
        self.target_url = f"https://{query.site}"
        self._llm_call_counts: dict[str, int] = {}

        self.openai_client = OpenAI(api_key=load_openai_api_key())
        self.session_storage_path = self.session_dir / SESSION_STORAGE_FILE
        self.cookies_path = self.session_dir / COOKIES_FILE
        self.login_form_memory_path = GLOBAL_LOGIN_INFO_FILE
        self._captured_login_fields: dict[tuple[str, str, str], dict[str, Any]] = {}

        self._playwright = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

        self.exploration_runs: list[dict[str, Any]] = []

    def _print_plan_command(self, command: PlanCommand) -> None:
        line = _plan_command_line(command)
        print(f"[plan] {line}", flush=True)

    def _print_exploration_step(
        self,
        *,
        step_idx: int,
        max_steps: int,
        message: str,
    ) -> None:
        print(f"[explore step {step_idx + 1}/{max_steps}] {message}", flush=True)

    def _next_llm_index(self, prefix: str) -> int:
        value = self._llm_call_counts.get(prefix, 0)
        self._llm_call_counts[prefix] = value + 1
        return value

    def _write_llm_call_log(
        self,
        prefix: str,
        request: Any,
        response: Any,
        *,
        index: int | None = None,
        error: Exception | None = None,
    ) -> None:
        if index is None:
            index = self._next_llm_index(prefix)
        payload = {
            "llm_call": prefix,
            "index": index,
            "provider": "openai",
            "request": request,
            "response": response,
        }
        if error is not None:
            payload["error"] = str(error)
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
        launch_kwargs = _launch_context_kwargs(browser=self.query.browser, headless=headless)
        if self.query.browser == CHROMIUM_BROWSER:
            self._context = await self._playwright.chromium.launch_persistent_context(
                str(self.session_dir),
                **launch_kwargs,
            )
        else:
            self._context = await self._playwright.firefox.launch_persistent_context(
                str(self.session_dir),
                **launch_kwargs,
            )
        human_login_injection_disabled = bool(self.query.nofill and not headless)
        if not human_login_injection_disabled:
            if not headless:
                await self._install_human_login_stealth_init_script()
            await self._install_session_storage_init_script()
        await self._restore_cookies()
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
        self._print_plan_command(command)

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

        if name in {"click", "type", "keypress", "wait", "vscroll"}:
            await _execute_plan_interaction(self.page, command, allow_text_entry=self.allow_text_entry)
            await self.recorder.capture(
                self.page,
                source=name,
                metadata={"plan_command": _plan_command_payload(command)},
            )
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
    allow_manual_login: bool = True,
) -> QueryOutcome:
    start = time.perf_counter()
    success = False
    skipped = False
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
            allow_manual_login=allow_manual_login,
        )
        body = asyncio.run(executor.run())
        headless = executor.headless
        help_used = executor.help_used
        stripped = body.lstrip()
        success = not stripped.startswith("FAIL") and not stripped.startswith("SKIP")
        if not body.strip():
            body = "FAIL\nNo answer text was produced by the plan."
            success = False
    except ManualLoginRequiredError as exc:
        body = f"SKIP\n{exc}"
        success = False
        skipped = True
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
        skipped=skipped,
        results_path=results_path,
        pertinent_text=pertinent,
        elapsed_ms=elapsed_ms,
        headless=headless,
        help_used=help_used,
    )


__all__ = [
    "GLOBAL_LOGIN_INFO_FILE",
    "LockRegistry",
    "ManualLoginRequiredError",
    "PlanExecutor",
    "QueryOutcome",
    "body_from_results_file",
    "run_query_execution",
    "write_results_file",
]

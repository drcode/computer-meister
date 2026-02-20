from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai import gemini_fast
from websites import WebsiteQuery


DEFAULT_EXPLORE_STEPS = 18
MAX_HISTORY_ATTEMPTS = 10


@dataclass(frozen=True)
class PlanCommand:
    name: str
    args: tuple[Any, ...] = ()


@dataclass(frozen=True)
class AttemptHistory:
    session_id: str
    plan_text: str
    result_text: str
    success: bool
    exploration_steps_text: str | None


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned.strip()


def parse_plan_line(line: str) -> PlanCommand:
    tokens = shlex.split(line, posix=True)
    if not tokens:
        raise ValueError("empty command")

    cmd = tokens[0]
    if cmd == "target_site":
        if len(tokens) != 2:
            raise ValueError("target_site expects one URL argument")
        return PlanCommand(cmd, (tokens[1],))

    if cmd in {"login_required", "enable_text_entry"}:
        if len(tokens) != 1:
            raise ValueError(f"{cmd} takes no arguments")
        return PlanCommand(cmd)

    if cmd == "explore_website_openai":
        if len(tokens) < 3:
            raise ValueError("explore_website_openai expects instruction and max_command_steps")
        try:
            steps = int(tokens[-1])
        except ValueError as exc:
            raise ValueError("explore_website_openai max_command_steps must be an integer") from exc
        instruction = " ".join(tokens[1:-1]).strip()
        if not instruction:
            raise ValueError("explore_website_openai instruction cannot be empty")
        return PlanCommand(cmd, (instruction, steps))

    if cmd == "click":
        if len(tokens) != 3:
            raise ValueError("click expects x y")
        return PlanCommand(cmd, (int(tokens[1]), int(tokens[2])))

    if cmd == "type":
        if len(tokens) < 2:
            raise ValueError("type expects text")
        return PlanCommand(cmd, (" ".join(tokens[1:]),))

    if cmd == "wait":
        if len(tokens) != 2:
            raise ValueError("wait expects milliseconds")
        return PlanCommand(cmd, (int(tokens[1]),))

    if cmd == "vscroll":
        if len(tokens) != 2:
            raise ValueError("vscroll expects a signed integer")
        return PlanCommand(cmd, (int(tokens[1]),))

    if cmd in {"answer_query_images", "answer_query_text"}:
        if len(tokens) < 2:
            raise ValueError(f"{cmd} expects instructions")
        return PlanCommand(cmd, (" ".join(tokens[1:]),))

    raise ValueError(f"unknown command: {cmd}")


def parse_plan_text(plan_text: str) -> list[PlanCommand]:
    commands: list[PlanCommand] = []
    for line_number, raw in enumerate(_strip_markdown_fences(plan_text).splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("//"):
            continue
        try:
            commands.append(parse_plan_line(line))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid plan line {line_number}: {line} ({exc})") from exc

    if not commands:
        raise ValueError("plan contains no commands")
    return commands


def render_plan(commands: list[PlanCommand]) -> str:
    lines: list[str] = []
    for cmd in commands:
        if cmd.name == "target_site":
            lines.append(f'target_site "{cmd.args[0]}"')
        elif cmd.name in {"login_required", "enable_text_entry"}:
            lines.append(cmd.name)
        elif cmd.name == "explore_website_openai":
            lines.append(f'explore_website_openai "{cmd.args[0]}" {int(cmd.args[1])}')
        elif cmd.name == "click":
            lines.append(f"click {int(cmd.args[0])} {int(cmd.args[1])}")
        elif cmd.name == "type":
            lines.append(f'type "{cmd.args[0]}"')
        elif cmd.name == "wait":
            lines.append(f"wait {int(cmd.args[0])}")
        elif cmd.name == "vscroll":
            lines.append(f"vscroll {int(cmd.args[0])}")
        elif cmd.name in {"answer_query_images", "answer_query_text"}:
            lines.append(f'{cmd.name} "{cmd.args[0]}"')
        else:
            raise ValueError(f"cannot render command: {cmd.name}")
    return "\n".join(lines) + "\n"


def _is_login_likely(query: WebsiteQuery) -> bool:
    site = query.site.lower()
    text = query.query.lower()
    login_sites = {
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "etrade.com",
        "fidelity.com",
        "bankofamerica.com",
        "chase.com",
        "gmail.com",
    }
    login_hints = [
        "notifications",
        "my ",
        "portfolio",
        "holdings",
        "cash balance",
        "account",
        "feed",
        "inbox",
    ]
    return site in login_sites or any(hint in text for hint in login_hints)


def _is_text_entry_likely(query: WebsiteQuery) -> bool:
    text = query.query.lower()
    hints = [
        "find the price",
        "price of",
        "ticker",
        "symbol",
        "search",
        "look up",
        "zip code",
        "city",
        "in ",
        "for ",
    ]
    return any(hint in text for hint in hints)


def _result_body(results_md: str) -> str:
    marker = "\n---\n"
    if marker in results_md:
        return results_md.split(marker, 1)[1].strip()
    return results_md.strip()


def _is_success_result(results_md: str) -> bool:
    body = _result_body(results_md)
    return not body.lstrip().startswith("FAIL")


def _trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32] + "\n...[truncated]..."


def load_attempt_history(query_dir: Path) -> list[AttemptHistory]:
    entries: list[tuple[int, AttemptHistory]] = []
    for plan_file in query_dir.glob("*.plan"):
        sid = plan_file.stem
        if not sid.isdigit():
            continue
        results_file = query_dir / f"{sid}_results.md"
        if not results_file.exists():
            continue

        plan_text = plan_file.read_text(encoding="utf-8")
        result_text = results_file.read_text(encoding="utf-8")
        exploration_path = query_dir / f"{sid}_artifacts" / "exploration_steps.json"
        exploration_text = None
        if exploration_path.exists():
            exploration_text = exploration_path.read_text(encoding="utf-8")

        entries.append(
            (
                int(sid),
                AttemptHistory(
                    session_id=sid,
                    plan_text=plan_text,
                    result_text=result_text,
                    success=_is_success_result(result_text),
                    exploration_steps_text=exploration_text,
                ),
            )
        )

    entries.sort(key=lambda item: item[0])
    return [item[1] for item in entries][-MAX_HISTORY_ATTEMPTS:]


def _make_initial_prompt(query: WebsiteQuery) -> str:
    needs_login = _is_login_likely(query)
    needs_text_entry = _is_text_entry_likely(query)

    return f"""Create a .plan file for this retrieval task.

SITE: {query.site}
QUERY: {query.query}

Allowed commands:
- target_site "url"
- login_required
- enable_text_entry
- explore_website_openai "exploration command" max_command_steps
- click x y
- type "text"
- wait ms
- vscroll amount
- answer_query_images "query instructions"
- answer_query_text "query instructions"

Requirements:
1) Output only the plan commands, one per line. No markdown fences.
2) First command must be target_site "https://{query.site}".
3) Since this is an initial plan, DO NOT use click/type/wait/vscroll.
4) Add login_required if likely needed.
5) Add enable_text_entry only if text entry is likely required.
6) Include one explore_website_openai command that describes the exploratory part.
7) Include one answer command (answer_query_text or answer_query_images) describing the answering part with precise required data points.

Hints for this query:
- likely_login_required: {needs_login}
- likely_text_entry_needed: {needs_text_entry}
"""


def _make_updated_prompt(query: WebsiteQuery, history: list[AttemptHistory]) -> str:
    history_payload: list[dict[str, Any]] = []
    for item in history:
        history_payload.append(
            {
                "session_id": item.session_id,
                "success": item.success,
                "plan": _trim_text(item.plan_text, 5000),
                "results_body": _trim_text(_result_body(item.result_text), 5000),
                "exploration_steps_json": _trim_text(item.exploration_steps_text or "", 10000),
            }
        )

    history_json = json.dumps(history_payload, indent=2)
    return f"""Create the next .plan file for this retrieval task.

SITE: {query.site}
QUERY: {query.query}

Allowed commands:
- target_site "url"
- login_required
- enable_text_entry
- explore_website_openai "exploration command" max_command_steps
- click x y
- type "text"
- wait ms
- vscroll amount
- answer_query_images "query instructions"
- answer_query_text "query instructions"

Previous attempts (latest up to 10) with plans/results/exploration steps:
{history_json}

Requirements:
1) Output only the plan commands, one per line. No markdown fences.
2) First command must be target_site "https://{query.site}".
3) Use failures to avoid repeated mistakes.
4) If prior attempts succeeded, reduce burden where possible (fewer steps, less user help).
5) You may replace exploration with concrete interactions when supported by successful exploration steps.
6) If typing is needed, include enable_text_entry before any type command.
7) End with one answer command that explicitly requests the final data points.
"""


def _host_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname:
        return parsed.hostname.lower()
    if "://" not in url:
        # Accept host-only strings.
        return url.split("/", 1)[0].lower()
    return ""


def _default_plan(query: WebsiteQuery, history_exists: bool) -> list[PlanCommand]:
    commands: list[PlanCommand] = [PlanCommand("target_site", (f"https://{query.site}",))]
    if _is_login_likely(query):
        commands.append(PlanCommand("login_required"))
    if _is_text_entry_likely(query):
        commands.append(PlanCommand("enable_text_entry"))
    steps = 12 if history_exists else DEFAULT_EXPLORE_STEPS
    commands.append(
        PlanCommand(
            "explore_website_openai",
            (
                f"Explore the site to locate the information needed to answer: {query.query}",
                steps,
            ),
        )
    )
    commands.append(
        PlanCommand(
            "answer_query_text",
            (
                f"Answer this exactly: {query.query}. If uncertain, begin with FAIL and explain what is missing.",
            ),
        )
    )
    return commands


def _normalize_plan(
    commands: list[PlanCommand],
    query: WebsiteQuery,
    *,
    history_exists: bool,
) -> list[PlanCommand]:
    out: list[PlanCommand] = [PlanCommand("target_site", (f"https://{query.site}",))]

    allow_manual_interactions = history_exists
    for cmd in commands:
        if cmd.name == "target_site":
            continue
        if not allow_manual_interactions and cmd.name in {"click", "type", "wait", "vscroll"}:
            continue
        out.append(cmd)

    if not any(cmd.name in {"answer_query_images", "answer_query_text"} for cmd in out):
        out.append(
            PlanCommand(
                "answer_query_text",
                (
                    f"Answer this exactly: {query.query}. If uncertain, begin with FAIL and explain what is missing.",
                ),
            )
        )

    # Keep answer commands at the end so collection happens after exploration/actions.
    non_answer = [cmd for cmd in out if cmd.name not in {"answer_query_images", "answer_query_text"}]
    answers = [cmd for cmd in out if cmd.name in {"answer_query_images", "answer_query_text"}]
    out = non_answer + answers

    has_action_before_answer = False
    for cmd in out:
        if cmd.name in {
            "explore_website_openai",
            "click",
            "type",
            "wait",
            "vscroll",
        }:
            has_action_before_answer = True
            break
    if not has_action_before_answer:
        out.insert(
            1,
            PlanCommand(
                "explore_website_openai",
                (
                    f"Explore the site to locate the information needed to answer: {query.query}",
                    DEFAULT_EXPLORE_STEPS,
                ),
            ),
        )

    needs_text_entry = any(cmd.name == "type" for cmd in out)
    if needs_text_entry and not any(cmd.name == "enable_text_entry" for cmd in out):
        insert_at = 1
        out.insert(insert_at, PlanCommand("enable_text_entry"))

    if not history_exists:
        if _is_login_likely(query) and not any(cmd.name == "login_required" for cmd in out):
            out.insert(1, PlanCommand("login_required"))
        if _is_text_entry_likely(query) and not any(cmd.name == "enable_text_entry" for cmd in out):
            out.insert(1, PlanCommand("enable_text_entry"))

    # Enforce target host match when model gives a conflicting host in exploratory plans.
    fixed: list[PlanCommand] = []
    for cmd in out:
        if cmd.name != "target_site":
            fixed.append(cmd)
            continue
        host = _host_from_url(str(cmd.args[0]))
        if host and host != query.site.lower():
            fixed.append(PlanCommand("target_site", (f"https://{query.site}",)))
        else:
            fixed.append(cmd)

    return fixed


def create_plan_for_query(query: WebsiteQuery, session_id: str, query_dir: Path) -> tuple[Path, list[PlanCommand], list[AttemptHistory]]:
    history = load_attempt_history(query_dir)

    system = (
        "You are a precise .plan author. Output only valid plan lines. "
        "Do not include markdown, explanations, or unknown commands."
    )
    prompt = _make_updated_prompt(query, history) if history else _make_initial_prompt(query)

    try:
        raw_plan = gemini_fast(
            prompt,
            system=system,
            temperature=0,
            max_output_tokens=1800,
        )
        commands = parse_plan_text(str(raw_plan))
        commands = _normalize_plan(commands, query, history_exists=bool(history))
    except Exception:
        commands = _default_plan(query, history_exists=bool(history))

    plan_text = render_plan(commands)
    plan_path = query_dir / f"{session_id}.plan"
    plan_path.write_text(plan_text, encoding="utf-8")
    return plan_path, commands, history

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai import GEMINI_FAST_MODEL, gemini_fast
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
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            redacted = _redact_image_data_url(value)
            return redacted if redacted is not None else value
        return value
    if isinstance(value, Path):
        return str(value)
    marker = id(value)
    if marker in _seen:
        return "<cycle>"
    _seen.add(marker)
    try:
        if isinstance(value, dict):
            return {
                str(key): _sanitize_for_log(nested, _seen=_seen)
                for key, nested in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [_sanitize_for_log(item, _seen=_seen) for item in value]
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
                return _sanitize_for_log(dumped, _seen=_seen)
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


def _write_llm_call_log(artifact_dir: Path, *, prompt: str, system: str, response: Any, error: Exception | None = None) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "llm_call": "create_prompt",
        "provider": "gemini_fast",
        "model": GEMINI_FAST_MODEL,
        "request": {
            "system": system,
            "prompt": prompt,
        },
        "response": response,
    }
    if error is not None:
        payload["error"] = str(error)
    (artifact_dir / "create_prompt.json").write_text(_dump_json(payload), encoding="utf-8")


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

    if cmd == "keypress":
        if len(tokens) < 2:
            raise ValueError("keypress expects at least one key")
        return PlanCommand(cmd, tuple(tokens[1:]))

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
        elif cmd.name == "keypress":
            keys = " ".join(shlex.quote(str(key)) for key in cmd.args)
            lines.append(f"keypress {keys}")
        elif cmd.name == "wait":
            lines.append(f"wait {int(cmd.args[0])}")
        elif cmd.name == "vscroll":
            lines.append(f"vscroll {int(cmd.args[0])}")
        elif cmd.name in {"answer_query_images", "answer_query_text"}:
            lines.append(f'{cmd.name} "{cmd.args[0]}"')
        else:
            raise ValueError(f"cannot render command: {cmd.name}")
    return "\n".join(lines) + "\n"


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


def _plan_file_format_docs() -> str:
    docs_path = Path(__file__).resolve().parent / "docs" / "plan_file_format.md"
    return docs_path.read_text(encoding="utf-8")


def _make_initial_prompt(query: WebsiteQuery) -> str:
    docs_text = _plan_file_format_docs()
    return f"""Create a .plan file for this retrieval task.
Output only the plan commands, one per line. No markdown fences.

SITE: {query.site}
QUERY: {query.query}

Plan file format documentation:
{docs_text}
"""


def _make_updated_prompt(query: WebsiteQuery, history: list[AttemptHistory]) -> str:
    docs_text = _plan_file_format_docs()
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
    guidance_lines: list[str] = []
    if any(not item.success for item in history):
        guidance_lines.append("Try to avoid the mistakes in the previous failures.")
    if any(item.success for item in history):
        guidance_lines.append("Try to reduce the burden from previous successful attempts (faster, fewer steps, less user help).")
    guidance_block = "\n".join(guidance_lines)
    guidance_text = f"{guidance_block}\n\n" if guidance_block else ""

    return f"""Create the next .plan file for this retrieval task.
Output only the plan commands, one per line. No markdown fences.
{guidance_text}SITE: {query.site}
QUERY: {query.query}

Plan file format documentation:
{docs_text}

Previous attempts (latest up to 10) with plans/results/exploration steps:
{history_json}
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

    for cmd in commands:
        if cmd.name == "target_site":
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
            "keypress",
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
    artifacts_dir = query_dir / f"{session_id}_artifacts"
    plan_error: Exception | None = None
    response_payload: Any = None

    try:
        response_payload = str(
            gemini_fast(
                prompt,
                system=system,
                temperature=0,
                max_output_tokens=1800,
            )
        )
        commands = parse_plan_text(response_payload)
        commands = _normalize_plan(commands, query, history_exists=bool(history))
    except Exception as exc:  # noqa: BLE001
        plan_error = exc
        response_payload = f"error: {exc}"
        commands = _default_plan(query, history_exists=bool(history))
    finally:
        _write_llm_call_log(
            artifacts_dir,
            prompt=prompt,
            system=system,
            response=response_payload,
            error=plan_error,
        )

    plan_text = render_plan(commands)
    plan_path = query_dir / f"{session_id}.plan"
    plan_path.write_text(plan_text, encoding="utf-8")
    return plan_path, commands, history

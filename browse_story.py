from __future__ import annotations

import html
import json
import os
import re
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from websites import WebsiteQuery, parse_websites_md


LLM_ORDER = {
    "create_plan": 0,
    "is_logged_in_check": 1,
    "exploration_loop": 2,
    "answer_query_text": 3,
    "answer_query_images": 3,
}


@dataclass(frozen=True)
class SessionInstance:
    session_id: str
    plan_path: Path
    results_path: Path
    artifacts_dir: Path
    success: bool | None


@dataclass(frozen=True)
class QueryHistory:
    section_id: str
    site: str
    number: int | None
    query_text: str | None
    query_dir: Path
    attempts: list[SessionInstance]


@dataclass(frozen=True)
class LlmLogEntry:
    name: str
    index: int | None
    path: Path
    provider: str | None
    model: str | None
    request: Any
    response: Any
    error: str | None


def _session_sort_key(session_id: str) -> tuple[int, int | str]:
    if session_id.isdigit():
        return (0, int(session_id))
    return (1, session_id)


def _artifact_index(path: Path) -> int:
    match = re.search(r"_(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    return -1


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_json_loads(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _try_parse_websites_lookup(websites_path: Path) -> dict[str, WebsiteQuery]:
    if not websites_path.exists():
        return {}
    try:
        queries = parse_websites_md(websites_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to parse {websites_path}: {exc}", flush=True)
        return {}
    return {query.dir_name: query for query in queries}


def _extract_target_site(plan_path: Path) -> str | None:
    if not plan_path.exists():
        return None
    text = _read_text(plan_path)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("target_site"):
            continue
        match = re.match(r'^target_site\s+"([^"]+)"\s*$', stripped)
        if not match:
            continue
        raw_url = match.group(1)
        parsed = urlparse(raw_url)
        if parsed.hostname:
            return parsed.hostname
        if "://" not in raw_url:
            return raw_url.split("/", 1)[0]
    return None


def _infer_number_from_dir_name(dir_name: str) -> int | None:
    match = re.search(r"_(\d+)$", dir_name)
    if not match:
        return None
    return int(match.group(1))


def _infer_site_from_dir_name(dir_name: str) -> str:
    no_suffix = re.sub(r"_\d+$", "", dir_name)
    return no_suffix.replace("_", ".")


def _parse_results_file(results_path: Path) -> tuple[dict[str, str], str]:
    if not results_path.exists():
        return {}, ""

    text = _read_text(results_path)
    marker = "\n---\n"
    if marker in text:
        head, body = text.split(marker, 1)
    else:
        head, body = "", text

    meta: dict[str, str] = {}
    for raw_line in head.splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body.strip()


def _discover_attempts(query_dir: Path) -> list[SessionInstance]:
    out: list[SessionInstance] = []
    for plan_path in query_dir.glob("*.plan"):
        session_id = plan_path.stem
        results_path = query_dir / f"{session_id}_results.md"
        artifacts_dir = query_dir / f"{session_id}_artifacts"
        if not results_path.exists() and not artifacts_dir.exists():
            continue

        success: bool | None = None
        if results_path.exists():
            _, body = _parse_results_file(results_path)
            success = bool(body) and not body.lstrip().startswith("FAIL")

        out.append(
            SessionInstance(
                session_id=session_id,
                plan_path=plan_path,
                results_path=results_path,
                artifacts_dir=artifacts_dir,
                success=success,
            )
        )

    out.sort(key=lambda item: _session_sort_key(item.session_id))
    return out


def discover_query_histories(site_data_dir: Path, websites_path: Path) -> list[QueryHistory]:
    lookup = _try_parse_websites_lookup(websites_path)
    histories: list[QueryHistory] = []

    if not site_data_dir.exists():
        return histories

    for query_dir in sorted(site_data_dir.iterdir()):
        if not query_dir.is_dir() or query_dir.name.endswith("_session"):
            continue

        attempts = _discover_attempts(query_dir)
        if not attempts:
            continue

        matched = lookup.get(query_dir.name)
        if matched is not None:
            site = matched.site
            number: int | None = matched.number
            query_text = matched.query
        else:
            number = _infer_number_from_dir_name(query_dir.name)
            site = _extract_target_site(attempts[-1].plan_path) or _infer_site_from_dir_name(query_dir.name)
            query_text = None

        section_id = f"{site} {number}" if number is not None else site
        histories.append(
            QueryHistory(
                section_id=section_id,
                site=site,
                number=number,
                query_text=query_text,
                query_dir=query_dir,
                attempts=attempts,
            )
        )

    histories.sort(key=lambda item: _session_sort_key(item.attempts[-1].session_id), reverse=True)
    return histories


def _session_label(session_id: str) -> str:
    if not session_id.isdigit():
        return session_id
    try:
        dt = datetime.fromtimestamp(int(session_id), tz=timezone.utc)
    except Exception:
        return session_id
    return f"{session_id} ({dt.strftime('%Y-%m-%d %H:%M:%S UTC')})"


def _status_label(success: bool | None) -> str:
    if success is None:
        return "unknown"
    return "success" if success else "fail"


def _pick_query(histories: list[QueryHistory]) -> QueryHistory:
    print("\nAvailable queries:", flush=True)
    for idx, history in enumerate(histories, start=1):
        latest = history.attempts[-1]
        query_desc = f" :: {history.query_text}" if history.query_text else ""
        print(
            f"  {idx}) {history.section_id}"
            f"  [{len(history.attempts)} instances, latest {_session_label(latest.session_id)}, {_status_label(latest.success)}]"
            f"{query_desc}",
            flush=True,
        )

    while True:
        raw = input("Select query (number or '<site> <number>') [default: 1]: ").strip()
        if not raw:
            return histories[0]

        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(histories):
                return histories[index - 1]

        normalized = raw.casefold()
        for history in histories:
            if normalized == history.section_id.casefold():
                return history

        print("Invalid query selection. Try again.", flush=True)


def _pick_instance(history: QueryHistory) -> SessionInstance:
    attempts_desc = list(reversed(history.attempts))

    print(f"\nInstances for {history.section_id}:", flush=True)
    for idx, instance in enumerate(attempts_desc, start=1):
        status = _status_label(instance.success)
        print(f"  {idx}) {_session_label(instance.session_id)} [{status}]", flush=True)

    default_instance = attempts_desc[0]
    prompt = f"Select instance (number or session id) [default: {default_instance.session_id}]: "

    while True:
        raw = input(prompt).strip()
        if not raw:
            return default_instance

        for instance in attempts_desc:
            if raw == instance.session_id:
                return instance

        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(attempts_desc):
                return attempts_desc[index - 1]

        print("Invalid instance selection. Try again.", flush=True)


def _relative_path(from_file: Path, to_path: Path) -> str:
    rel = os.path.relpath(to_path, start=from_file.parent)
    return rel.replace(os.sep, "/")


def _pretty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _html_pre(value: Any) -> str:
    return f"<pre>{html.escape(_pretty(value))}</pre>"


def _prompt_pre(prompt_value: Any) -> str:
    if prompt_value is None:
        return ""
    if not isinstance(prompt_value, str):
        prompt_value = _pretty(prompt_value)
    return f"<pre class=\"prompt\">{html.escape(prompt_value)}</pre>"


def _parse_name_and_index(path: Path) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_(\d+)$", path.stem)
    if match:
        return match.group(1), int(match.group(2))
    return path.stem, None


def _load_llm_logs(artifacts_dir: Path) -> list[LlmLogEntry]:
    entries: list[LlmLogEntry] = []

    if not artifacts_dir.exists():
        return entries

    for path in sorted(artifacts_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "exploration_steps.json":
            continue
        if path.name.startswith("page_") or path.name.startswith("screenshot_"):
            continue
        if path.suffix != ".json":
            continue

        name_from_file, index_from_file = _parse_name_and_index(path)
        payload = _safe_json_loads(_read_text(path))
        if not isinstance(payload, dict):
            entries.append(
                LlmLogEntry(
                    name=name_from_file,
                    index=index_from_file,
                    path=path,
                    provider=None,
                    model=None,
                    request=None,
                    response=f"Invalid JSON payload in {path.name}",
                    error=None,
                )
            )
            continue

        name = str(payload.get("llm_call") or name_from_file)
        index = payload.get("index") if isinstance(payload.get("index"), int) else index_from_file
        provider = payload.get("provider") if isinstance(payload.get("provider"), str) else None
        model = payload.get("model") if isinstance(payload.get("model"), str) else None
        request = payload.get("request")
        response = payload.get("response")
        error = payload.get("error") if isinstance(payload.get("error"), str) else None

        if model is None and isinstance(request, dict) and isinstance(request.get("model"), str):
            model = request.get("model")

        entries.append(
            LlmLogEntry(
                name=name,
                index=index,
                path=path,
                provider=provider,
                model=model,
                request=request,
                response=response,
                error=error,
            )
        )

    entries.sort(
        key=lambda item: (
            LLM_ORDER.get(item.name, 9),
            item.index if item.index is not None else -1,
            item.path.name,
        )
    )
    return entries


def _load_exploration_runs(artifacts_dir: Path) -> list[dict[str, Any]]:
    path = artifacts_dir / "exploration_steps.json"
    if not path.exists():
        return []

    payload = _safe_json_loads(_read_text(path))
    if not isinstance(payload, dict):
        return []

    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for run in runs:
        if isinstance(run, dict):
            cleaned.append(run)
    return cleaned


def _response_summary(log: LlmLogEntry) -> str | None:
    if log.error:
        return f"ERROR: {log.error.splitlines()[0][:280]}"

    if isinstance(log.response, str):
        text = log.response.strip()
        if text:
            return text.splitlines()[0][:300]

    if isinstance(log.response, dict):
        for key in ("output_text", "text", "response"):
            value = log.response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().splitlines()[0][:300]

    return None


def _request_image_artifacts(log: LlmLogEntry) -> list[str]:
    request = log.request if isinstance(log.request, dict) else None
    if request is None:
        return []

    names: list[str] = []
    single = request.get("input_image_artifact")
    if isinstance(single, str) and single.strip():
        names.append(single.strip())

    multiple = request.get("input_image_artifacts")
    if isinstance(multiple, list):
        for item in multiple:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())

    return names


def _request_inline_image_descriptions(log: LlmLogEntry) -> list[str]:
    request = log.request if isinstance(log.request, dict) else None
    if request is None:
        return []

    input_items = request.get("input")
    if not isinstance(input_items, list):
        return []

    out: list[str] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        content_items = item.get("content")
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if not isinstance(content, dict):
                continue
            if str(content.get("type", "")) != "input_image":
                continue
            image_url = content.get("image_url")
            if isinstance(image_url, str) and image_url.strip():
                out.append(image_url.strip())
    return out


def _render_llm_entry(log: LlmLogEntry, *, artifacts_dir: Path, output_file: Path) -> str:
    title = log.name if log.index is None else f"{log.name} #{log.index}"
    source_line = f"{log.provider or 'unknown provider'}"
    if log.model:
        source_line += f" | model: {log.model}"

    parts = [
        '<article class="card">',
        f"<h3>{html.escape(title)}</h3>",
        f"<p class=\"meta\">{html.escape(source_line)} | file: {html.escape(log.path.name)}</p>",
    ]

    summary = _response_summary(log)
    if summary:
        parts.append(f"<p><strong>Response summary:</strong> {html.escape(summary)}</p>")

    image_names = _request_image_artifacts(log)
    inline_images = _request_inline_image_descriptions(log)
    if image_names or inline_images:
        parts.append("<p><strong>Input images:</strong></p>")
        for image_name in image_names:
            image_path = artifacts_dir / image_name
            if not image_path.exists():
                parts.append(f"<p class=\"meta\">missing image artifact: {html.escape(image_name)}</p>")
                continue
            rel = _relative_path(output_file, image_path)
            safe_rel = html.escape(rel, quote=True)
            safe_name = html.escape(image_name)
            parts.append(f"<p class=\"meta\">{safe_name}</p>")
            parts.append(f'<a href="{safe_rel}"><img src="{safe_rel}" loading="lazy" /></a>')
        if inline_images:
            parts.append("<details><summary>Input image payloads</summary>")
            parts.append(_html_pre(inline_images))
            parts.append("</details>")

    if log.request is not None:
        request_obj: Any = log.request
        if isinstance(log.request, dict) and "prompt" in log.request:
            prompt_value = log.request["prompt"]
            request_obj = dict(log.request)
            request_obj.pop("prompt", None)
            parts.append("<details><summary>Prompt</summary>")
            parts.append(_prompt_pre(prompt_value))
            parts.append("</details>")

        parts.append("<details><summary>Request</summary>")
        parts.append(_html_pre(request_obj))
        parts.append("</details>")

    if log.error:
        parts.append("<details><summary>Error</summary>")
        parts.append(_html_pre(log.error))
        parts.append("</details>")

    if log.response is not None:
        parts.append("<details><summary>Response</summary>")
        parts.append(_html_pre(log.response))
        parts.append("</details>")

    parts.append("</article>")
    return "\n".join(parts)


def _render_exploration(runs: list[dict[str, Any]], artifacts_dir: Path, output_file: Path) -> str:
    if not runs:
        return '<p class="muted">No exploration_steps.json was found for this instance.</p>'

    blocks: list[str] = []
    for idx, run in enumerate(runs, start=1):
        instruction = str(run.get("instruction", ""))
        final_text = str(run.get("final_text", ""))
        max_steps = run.get("max_command_steps")
        allow_text = run.get("allow_text_entry")
        steps = run.get("steps") if isinstance(run.get("steps"), list) else []

        blocks.append('<article class="card">')
        blocks.append(f"<h3>Exploration run {idx}</h3>")
        blocks.append(
            "<p class=\"meta\">"
            f"max steps: {html.escape(str(max_steps))}"
            f" | text entry allowed: {html.escape(str(allow_text))}"
            "</p>"
        )
        blocks.append(f"<p><strong>Instruction:</strong> {html.escape(instruction)}</p>")

        if final_text:
            blocks.append("<details><summary>Final exploration summary</summary>")
            blocks.append(_html_pre(final_text))
            blocks.append("</details>")

        if steps:
            blocks.append('<div class="steps">')
            for step in steps:
                if not isinstance(step, dict):
                    continue
                step_index = step.get("step")
                plan_command = step.get("plan_command") if isinstance(step.get("plan_command"), dict) else {}
                plan_line = str(plan_command.get("line", "")) if plan_command else ""
                command_name = str(plan_command.get("name", "")) if plan_command else ""
                computer_action = step.get("computer_action") if isinstance(step.get("computer_action"), dict) else {}
                computer_action_type = str(computer_action.get("type", "")) if computer_action else ""
                error = step.get("error")
                artifact = step.get("artifact") if isinstance(step.get("artifact"), dict) else {}
                screenshot_name = artifact.get("screenshot") if isinstance(artifact.get("screenshot"), str) else None
                page_name = artifact.get("page_html") if isinstance(artifact.get("page_html"), str) else None

                blocks.append('<div class="step">')
                label = command_name or computer_action_type
                blocks.append(
                    f"<h4>Step {html.escape(str(step_index))}"
                    f" {html.escape(label) if label else ''}</h4>"
                )

                if screenshot_name:
                    screenshot_path = artifacts_dir / screenshot_name
                    if screenshot_path.exists():
                        rel = _relative_path(output_file, screenshot_path)
                        blocks.append(f'<a href="{html.escape(rel)}"><img src="{html.escape(rel)}" loading="lazy" /></a>')

                if page_name:
                    blocks.append(f"<p class=\"meta\">page snapshot: {html.escape(page_name)}</p>")

                if plan_command:
                    blocks.append("<details><summary>Plan command</summary>")
                    if plan_line:
                        blocks.append(_html_pre(plan_line))
                    else:
                        blocks.append(_html_pre(plan_command))
                    blocks.append("</details>")
                elif computer_action:
                    blocks.append("<details><summary>Computer action</summary>")
                    blocks.append(_html_pre(computer_action))
                    blocks.append("</details>")

                if error:
                    blocks.append("<details><summary>Action error</summary>")
                    blocks.append(_html_pre(error))
                    blocks.append("</details>")

                blocks.append("</div>")
            blocks.append("</div>")

        blocks.append("</article>")

    return "\n".join(blocks)


def _render_screenshot_gallery(runs: list[dict[str, Any]], artifacts_dir: Path, output_file: Path) -> str:
    screenshots = sorted(artifacts_dir.glob("screenshot_*.png"), key=_artifact_index)
    if not screenshots:
        return '<p class="muted">No screenshots found.</p>'

    notes: dict[str, str] = {}
    for run_index, run in enumerate(runs, start=1):
        steps = run.get("steps") if isinstance(run.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            artifact = step.get("artifact") if isinstance(step.get("artifact"), dict) else {}
            screenshot_name = artifact.get("screenshot") if isinstance(artifact.get("screenshot"), str) else None
            if not screenshot_name:
                continue
            plan_command = step.get("plan_command") if isinstance(step.get("plan_command"), dict) else {}
            if plan_command:
                action_type = str(plan_command.get("name", ""))
            else:
                computer_action = step.get("computer_action") if isinstance(step.get("computer_action"), dict) else {}
                action_type = str(computer_action.get("type", "")) if computer_action else ""
            step_idx = step.get("step")
            notes[screenshot_name] = f"run {run_index}, step {step_idx}, action {action_type}".strip()

    items: list[str] = ['<div class="gallery-shell">']
    items.append('<div id="screenshot-gallery" class="gallery" role="list" aria-label="Screenshots">')
    for index, screenshot in enumerate(screenshots):
        rel = _relative_path(output_file, screenshot)
        caption = notes.get(screenshot.name, screenshot.name)
        safe_rel = html.escape(rel, quote=True)
        safe_caption = html.escape(caption, quote=True)
        items.append(f'<button type="button" class="card gallery-thumb" role="listitem" data-index="{index}" data-src="{safe_rel}" data-caption="{safe_caption}">')
        items.append(f'<img src="{safe_rel}" alt="{safe_caption}" loading="lazy" />')
        items.append(f"<span>{safe_caption}</span>")
        items.append("</button>")
    items.append("</div>")
    items.extend(
        [
            '<div id="screenshot-viewer" class="gallery-viewer" aria-live="polite" hidden>',
            '  <div class="viewer-top">',
            '    <span id="screenshot-viewer-index" class="viewer-meta"></span>',
            '    <button id="screenshot-viewer-close" type="button">Close</button>',
            "  </div>",
            '  <div class="viewer-frame">',
            '    <button id="screenshot-viewer-prev" type="button" class="viewer-control">Prev</button>',
            '    <img id="screenshot-viewer-image" src="" alt="Selected screenshot" />',
            '    <button id="screenshot-viewer-next" type="button" class="viewer-control">Next</button>',
            "  </div>",
            '  <p id="screenshot-viewer-caption" class="viewer-caption"></p>',
            "</div>",
        ]
    )
    items.append("</div>")
    return "\n".join(items)


def _build_story_html(history: QueryHistory, instance: SessionInstance, output_file: Path) -> str:
    plan_text = _read_text(instance.plan_path) if instance.plan_path.exists() else ""
    results_meta, results_body = _parse_results_file(instance.results_path)
    llm_logs = _load_llm_logs(instance.artifacts_dir)
    runs = _load_exploration_runs(instance.artifacts_dir)

    create_logs = [log for log in llm_logs if log.name == "create_plan"]
    non_create_logs = [log for log in llm_logs if log.name != "create_plan"]

    meta_lines: list[str] = []
    for key in ("time", "headless", "help"):
        if key in results_meta:
            meta_lines.append(f"{key}: {results_meta[key]}")

    title = f"Query Story: {history.section_id} / {instance.session_id}"

    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        f"  <title>{html.escape(title)}</title>",
        "  <style>",
        "    :root { --bg: #f6f7fb; --card: #ffffff; --text: #1f2430; --muted: #5f6b7a; --border: #d8deea; --accent: #224e87; }",
        "    * { box-sizing: border-box; }",
        "    body { margin: 0; padding: 24px; font-family: 'Segoe UI', Tahoma, sans-serif; background: radial-gradient(circle at top right, #e8effa 0%, var(--bg) 35%, #f8f5ef 100%); color: var(--text); }",
        "    main { max-width: 1200px; margin: 0 auto; }",
        "    h1, h2, h3, h4 { margin: 0 0 10px; }",
        "    section { margin-top: 18px; }",
        "    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(18, 35, 54, 0.06); }",
        "    .meta { color: var(--muted); font-size: 0.94rem; }",
        "    .muted { color: var(--muted); }",
        "    pre { margin: 8px 0 0; background: #0f1724; color: #f6f8fc; padding: 12px; border-radius: 10px; overflow: auto; white-space: pre-wrap; word-break: break-word; }",
        "    details { margin-top: 10px; }",
        "    summary { cursor: pointer; color: var(--accent); font-weight: 600; }",
        "    .steps { display: grid; gap: 12px; margin-top: 12px; }",
        "    .step { background: #f9fbff; border: 1px solid var(--border); border-radius: 10px; padding: 10px; }",
        "    img { max-width: 100%; border-radius: 8px; border: 1px solid #cfd7e4; }",
        "    .gallery { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }",
        "    figure { margin: 0; }",
        "    figcaption { margin-top: 8px; color: var(--muted); font-size: 0.9rem; }",
        "    .gallery-shell { display: grid; gap: 12px; }",
        "    .gallery-thumb { text-align: left; border: 1px solid var(--border); border-radius: 10px; padding: 8px; background: #f9fbff; color: inherit; cursor: pointer; }",
        "    .gallery-thumb:hover, .gallery-thumb:focus-visible { border-color: var(--accent); outline: none; }",
        "    .gallery-thumb span { display: block; margin-top: 6px; color: var(--muted); font-size: 0.9rem; }",
        "    .gallery-viewer { border: 1px solid var(--border); border-radius: 12px; background: #ffffff; padding: 12px; box-shadow: 0 8px 20px rgba(12, 22, 42, 0.1); }",
        "    .gallery-viewer[hidden] { display: none; }",
        "    .viewer-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }",
        "    .viewer-frame { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 8px; }",
        "    .viewer-control { background: #eef3fc; color: var(--text); border: 1px solid var(--border); border-radius: 999px; min-width: 70px; padding: 10px 12px; font-weight: 600; }",
        "    .viewer-control:hover, .viewer-control:focus-visible { background: #e2ecff; outline: none; }",
        "    .viewer-caption { margin: 8px 0 0; color: var(--muted); }",
        "    .viewer-meta { color: var(--muted); font-size: 0.9rem; }",
        "  </style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{html.escape(title)}</h1>",
        '<section class="card">',
        f"<p><strong>Query:</strong> {html.escape(history.section_id)}</p>",
        f"<p><strong>Instance:</strong> {html.escape(_session_label(instance.session_id))}</p>",
        f"<p><strong>Query text:</strong> {html.escape(history.query_text or 'unknown')}</p>",
        f"<p><strong>Artifacts:</strong> {html.escape(str(instance.artifacts_dir))}</p>",
        "</section>",
        "<section>",
        "<h2>Outcome</h2>",
        '<article class="card">',
    ]

    if meta_lines:
        parts.append(f"<p class=\"meta\">{' | '.join(html.escape(line) for line in meta_lines)}</p>")
    else:
        parts.append('<p class="meta">No results metadata found.</p>')
    parts.append(_html_pre(results_body or "No results body found."))
    parts.extend(
        [
            "</article>",
            "</section>",
            "<section>",
            "<h2>Plan</h2>",
            '<article class="card">',
            _html_pre(plan_text or "No .plan file found."),
            "</article>",
            "</section>",
            "<section>",
            "<h2>How The Plan Was Made</h2>",
        ]
    )

    if create_logs:
        for log in create_logs:
            parts.append(_render_llm_entry(log, artifacts_dir=instance.artifacts_dir, output_file=output_file))
    else:
        parts.append('<p class="muted">No create_plan log found.</p>')

    parts.extend(
        [
            "</section>",
            "<section>",
            "<h2>How The Site Was Browsed</h2>",
            _render_exploration(runs, instance.artifacts_dir, output_file),
            "</section>",
            "<section>",
            "<h2>LLM Prompts And Results</h2>",
        ]
    )

    if non_create_logs:
        for log in non_create_logs:
            parts.append(_render_llm_entry(log, artifacts_dir=instance.artifacts_dir, output_file=output_file))
    else:
        parts.append('<p class="muted">No additional LLM logs found.</p>')

    parts.extend(
        [
            "</section>",
            "<section>",
            "<h2>All Screenshots</h2>",
            _render_screenshot_gallery(runs, instance.artifacts_dir, output_file),
            "</section>",
            "</main>",
            "<script>",
            "  const gallery = document.querySelector('#screenshot-gallery');",
            "  const thumbs = gallery ? Array.from(gallery.querySelectorAll('.gallery-thumb')) : [];",
            "  const viewer = document.getElementById('screenshot-viewer');",
            "  const viewerImage = document.getElementById('screenshot-viewer-image');",
            "  const viewerCaption = document.getElementById('screenshot-viewer-caption');",
            "  const viewerIndex = document.getElementById('screenshot-viewer-index');",
            "  const prevButton = document.getElementById('screenshot-viewer-prev');",
            "  const nextButton = document.getElementById('screenshot-viewer-next');",
            "  const closeButton = document.getElementById('screenshot-viewer-close');",
            "  let currentIndex = 0;",
            "",
            "  function openViewer(index) {",
            "    if (!viewer || !thumbs.length) {",
            "      return;",
            "    }",
            "    currentIndex = ((index % thumbs.length) + thumbs.length) % thumbs.length;",
            "    const thumb = thumbs[currentIndex];",
            "    const src = thumb.dataset.src || '';",
            "    const caption = thumb.dataset.caption || '';",
            "    viewerImage.src = src;",
            "    viewerImage.alt = caption || `Screenshot ${currentIndex + 1}`;",
            "    viewerCaption.textContent = caption || `Screenshot ${currentIndex + 1}`;",
            "    viewerIndex.textContent = `${currentIndex + 1} / ${thumbs.length}`;",
            "    viewer.hidden = false;",
            "  }",
            "",
            "  function move(delta) {",
            "    if (!viewer || viewer.hidden) {",
            "      return;",
            "    }",
            "    openViewer(currentIndex + delta);",
            "  }",
            "",
            "  function closeViewer() {",
            "    if (!viewer) {",
            "      return;",
            "    }",
            "    viewer.hidden = true;",
            "  }",
            "",
            "  thumbs.forEach((thumb, index) => {",
            "    thumb.addEventListener('click', () => openViewer(index));",
            "  });",
            "",
            "  if (prevButton) {",
            "    prevButton.addEventListener('click', () => move(-1));",
            "  }",
            "  if (nextButton) {",
            "    nextButton.addEventListener('click', () => move(1));",
            "  }",
            "  if (closeButton) {",
            "    closeButton.addEventListener('click', closeViewer);",
            "  }",
            "",
            "  window.addEventListener('keydown', (event) => {",
            "    if (!viewer || viewer.hidden) {",
            "      return;",
            "    }",
            "    if (event.key === 'Escape') {",
            "      closeViewer();",
            "      return;",
            "    }",
            "    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft' || event.key.toLowerCase() === 'n') {",
            "      event.preventDefault();",
            "      if (event.key === 'ArrowLeft') {",
            "        move(-1);",
            "      } else {",
            "        move(1);",
            "      }",
            "    }",
            "  });",
            "</script>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(parts)


def _write_story_html(history: QueryHistory, instance: SessionInstance, browser_data_dir: Path) -> Path:
    browser_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = browser_data_dir / f"story_{history.query_dir.name}_{instance.session_id}.html"
    html_text = _build_story_html(history, instance, output_path)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def run_browse_story_mode(*, websites_path: Path) -> None:
    site_data_dir = Path("site_data")
    browser_data_dir = Path("browser_data")

    histories = discover_query_histories(site_data_dir=site_data_dir, websites_path=websites_path)
    if not histories:
        print("No query artifact sessions found under site_data/", flush=True)
        return

    history = _pick_query(histories)
    instance = _pick_instance(history)

    if not instance.artifacts_dir.exists():
        print(f"Artifact directory not found: {instance.artifacts_dir}", flush=True)
        return

    output_path = _write_story_html(history, instance, browser_data_dir)
    print(f"Generated story HTML: {output_path}", flush=True)

    opened = webbrowser.open(output_path.resolve().as_uri())

    if opened:
        print("Opened story in default browser.", flush=True)
    else:
        print(f"Could not auto-open browser. Open this file manually: {output_path.resolve()}", flush=True)

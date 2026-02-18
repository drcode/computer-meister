from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI

OPENAI_SMART_MODEL = "gpt-5.2"
OPENAI_SMART_EFFORT = "high"


def _find_key_file(filename: str, *, extra_paths: Iterable[Path] | None = None) -> str | None:
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


@functools.lru_cache(maxsize=2)
def _openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _extract_openai_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return "".join(chunk.get("text", "") for chunk in content if isinstance(chunk, dict)).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def _normalize_messages(
    prompt: str | None,
    system: str | None,
    messages: Sequence[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if messages:
        combined = [dict(m) for m in messages]
        if system:
            return [{"role": "system", "content": system}, *combined]
        return combined
    if prompt is None:
        raise ValueError("prompt is required when messages is not provided")
    if system:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]


def _openai_text(
    prompt: str | None,
    *,
    model: str,
    system: str | None = None,
    messages: Sequence[dict[str, str]] | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    client: OpenAI | None = None,
) -> str:
    key = (api_key or "").strip() or load_openai_api_key()
    client = client or _openai_client(key)

    messages_out = _normalize_messages(prompt, system, messages)
    kwargs: dict[str, Any] = {"model": model, "messages": messages_out}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_output_tokens is not None:
        kwargs["max_completion_tokens"] = int(max_output_tokens)
    if temperature is not None:
        kwargs["temperature"] = float(temperature)

    response = client.chat.completions.create(**kwargs)
    return _extract_openai_message_text(response.choices[0].message)


def openai_smart(
    prompt: str | None = None,
    *,
    system: str | None = None,
    messages: Sequence[dict[str, str]] | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    client: OpenAI | None = None,
) -> str:
    effort = reasoning_effort or OPENAI_SMART_EFFORT
    return _openai_text(
        prompt,
        model=OPENAI_SMART_MODEL,
        system=system,
        messages=messages,
        reasoning_effort=effort,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        api_key=api_key,
        client=client,
    )

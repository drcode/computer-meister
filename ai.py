from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from openai import OpenAI

OPENAI_MODEL = "gpt-5.2"
GEMINI_FAST_MODEL = "gemini-3-flash-preview"


def _find_key_file(filename: str, *, extra_paths: Iterable[Path] | None = None) -> str | None:
    candidates: list[Path] = []
    if extra_paths:
        candidates.extend(extra_paths)
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in [base] + list(base.parents):
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


def _extract_openai_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text.strip()
    return ""


def _response_format_expects_json(response_format: Any) -> bool:
    if response_format is None:
        return False
    if isinstance(response_format, dict):
        fmt_type = response_format.get("type")
    else:
        fmt_type = getattr(response_format, "type", None)
    return fmt_type in ("json_schema", "json_object")


def _parse_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty JSON response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        return json.loads(cleaned[obj_start : obj_end + 1])

    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        return json.loads(cleaned[arr_start : arr_end + 1])

    raise ValueError(f"could not parse JSON from response: {raw[:200]}")


def _normalize_messages(
    prompt: str | None,
    system: str | None,
    messages: Sequence[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
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


def _extract_user_prompt(prompt: str | None, messages: Sequence[dict[str, Any]] | None) -> str:
    if prompt is not None:
        return prompt
    if not messages:
        return ""
    parts: list[str] = []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
    return "\n\n".join(parts)


def _emit_show_data(user_prompt: str, result: Any) -> None:
    label = "[ai.show_data] "
    _print_with_label(label, "== PROMPT ==")
    _print_with_label(label, user_prompt)
    _print_with_label(label, "== RESPONSE ==")
    _print_with_label(label, result)


def _print_with_label(label: str, content: Any) -> None:
    text = "" if content is None else str(content)
    lines = text.splitlines()
    if not lines:
        print(f"{label}")
        return
    for line in lines:
        print(f"{label}{line}")


def _openai_text(
    prompt: str | None,
    *,
    model: str,
    system: str | None = None,
    messages: Sequence[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    client: OpenAI | None = None,
    return_response: bool = False,
    show_data: bool = False,
) -> Any:
    key = (api_key or "").strip() or load_openai_api_key()
    client = client or _openai_client(key)

    if tools or tool_choice:
        if messages:
            raise ValueError("messages are not supported when tools are provided")
        if prompt is None:
            raise ValueError("prompt is required when tools are provided")
        kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt,
        }
        if system:
            kwargs["instructions"] = system
        if tools:
            kwargs["tools"] = list(tools)
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if response_format is not None:
            kwargs["text"] = {"format": response_format}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        response = client.responses.create(**kwargs)
        if return_response:
            if show_data:
                _emit_show_data(prompt or "", response)
            return response
        text = _extract_openai_output_text(response)
        if _response_format_expects_json(response_format):
            parsed = _parse_json(text)
            if show_data:
                _emit_show_data(prompt or "", parsed)
            return parsed
        if show_data:
            _emit_show_data(prompt or "", text)
        return text

    messages_out = _normalize_messages(prompt, system, messages)
    kwargs = {"model": model, "messages": messages_out}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_output_tokens is not None:
        kwargs["max_completion_tokens"] = int(max_output_tokens)
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    response = client.chat.completions.create(**kwargs)
    if return_response:
        if show_data:
            user_prompt = _extract_user_prompt(prompt, messages)
            _emit_show_data(user_prompt, response)
        return response
    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        if show_data:
            user_prompt = _extract_user_prompt(prompt, messages)
            _emit_show_data(user_prompt, parsed)
        return parsed
    text = _extract_openai_message_text(message)
    if _response_format_expects_json(response_format):
        parsed = _parse_json(text)
        if show_data:
            user_prompt = _extract_user_prompt(prompt, messages)
            _emit_show_data(user_prompt, parsed)
        return parsed
    if show_data:
        user_prompt = _extract_user_prompt(prompt, messages)
        _emit_show_data(user_prompt, text)
    return text


def openai(
    prompt: str | None = None,
    *,
    system: str | None = None,
    messages: Sequence[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    tools: Sequence[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    client: OpenAI | None = None,
    return_response: bool = False,
    show_data: bool = False,
) -> Any:
    return _openai_text(
        prompt,
        model=OPENAI_MODEL,
        system=system,
        messages=messages,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        api_key=api_key,
        client=client,
        return_response=return_response,
        show_data=show_data,
    )


def load_gemini_api_key() -> str:
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY", "GENAI_API_KEY"):
        v = os.environ.get(env_name)
        if v and v.strip():
            return v.strip()

    candidates: list[Path] = []
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in [base] + list(base.parents):
            candidates.append(parent / "google_api_key.txt")

    for path in candidates:
        try:
            if path.exists():
                key = path.read_text(encoding="utf-8").strip()
                if key:
                    return key
        except Exception:
            continue

    raise RuntimeError(
        "Gemini API key not found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) or place google_api_key.txt in/above the working directory."
    )


def import_genai() -> tuple[Any, Any]:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        return genai, types
    except Exception as exc:
        raise RuntimeError("google-genai is required. Install it with: pip install google-genai") from exc


@functools.lru_cache(maxsize=2)
def _gemini_client(api_key: str) -> Any:
    genai, _types = import_genai()
    return genai.Client(api_key=api_key)


def _build_gemini_config(
    types: Any,
    *,
    system: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    response_mime_type: str | None = None,
    response_schema: Any | None = None,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
) -> Any | None:
    config_kwargs: dict[str, Any] = {}
    if system:
        config_kwargs["system_instruction"] = system
    if temperature is not None:
        config_kwargs["temperature"] = float(temperature)
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = int(max_output_tokens)
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if response_schema is not None:
        config_kwargs["response_schema"] = response_schema
    if thinking_budget is not None or thinking_level is not None:
        thinking_kwargs: dict[str, Any] = {}
        if thinking_budget is not None:
            thinking_kwargs["thinking_budget"] = int(thinking_budget)
        if thinking_level is not None:
            raw_level = str(thinking_level).strip()
            level_name = raw_level.upper()
            level_enum = getattr(types, "ThinkingLevel", None)
            if level_enum is not None and hasattr(level_enum, level_name):
                thinking_kwargs["thinking_level"] = getattr(level_enum, level_name)
            else:
                thinking_kwargs["thinking_level"] = level_name
        config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)
    if not config_kwargs:
        return None
    return types.GenerateContentConfig(**config_kwargs)


def _gemini_extract_text(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        text_chunks: list[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                text_chunks.append(part_text)
        if text_chunks:
            return "\n".join(text_chunks).strip()
    return ""


def _gemini_text(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    response_mime_type: str | None = None,
    response_schema: Any | None = None,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
    api_key: str | None = None,
    return_response: bool = False,
    show_data: bool = False,
) -> Any:
    _, types = import_genai()
    key = (api_key or "").strip() or load_gemini_api_key()
    client = _gemini_client(key)
    config = _build_gemini_config(
        types,
        system=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        thinking_budget=thinking_budget,
        thinking_level=thinking_level,
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    if return_response:
        if show_data:
            _emit_show_data(prompt, response)
        return response
    text = _gemini_extract_text(response)
    if response_schema is not None or response_mime_type == "application/json":
        parsed = _parse_json(text)
        if show_data:
            _emit_show_data(prompt, parsed)
        return parsed
    if show_data:
        _emit_show_data(prompt, text)
    return text


def gemini_fast(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    response_mime_type: str | None = None,
    response_schema: Any | None = None,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
    api_key: str | None = None,
    return_response: bool = False,
    show_data: bool = False,
) -> Any:
    level = "medium" if thinking_level is None else thinking_level
    return _gemini_text(
        prompt,
        model=GEMINI_FAST_MODEL,
        system=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        thinking_budget=thinking_budget,
        thinking_level=level,
        api_key=api_key,
        return_response=return_response,
        show_data=show_data,
    )

from __future__ import annotations

import base64
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from execution_common import (
    COMPUTER_USE_MODEL,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    _plan_command_line,
    _plan_command_payload,
    _safe_goto,
)
from execution_exploration import _computer_action_to_plan_command, _execute_plan_interaction


AUTOLOGIN_STATS_FILE = Path("site_data") / "autologin_stats.json"
AUTOLOGIN_MAX_ACTION_STEPS = 24
AUTOLOGIN_ATTEMPTS_FILE = "autologin_attempts.json"
AUTOLOGIN_BACKOFF_ENABLED = False
_AUTOLOGIN_STATS_LOCK = threading.Lock()


class AutologinMixin:
    def _append_autologin_attempt_record(self, record: dict[str, Any]) -> None:
        path = self.artifact_dir / AUTOLOGIN_ATTEMPTS_FILE
        payload: dict[str, Any] = {"runs": []}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                    payload = existing
            except Exception:
                payload = {"runs": []}

        payload.setdefault("runs", [])
        if isinstance(payload["runs"], list):
            payload["runs"].append(record)
        else:
            payload["runs"] = [record]
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _autologin_site_key(self) -> str:
        site = self._normalize_site_key(self.query.site)
        if site:
            return site
        parsed = urlparse(self.target_url)
        host = (parsed.hostname or "").strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _autologin_site_matches(self, site_key: str, candidate_host: str) -> bool:
        return (
            site_key == candidate_host
            or site_key.endswith(f".{candidate_host}")
            or candidate_host.endswith(f".{site_key}")
        )

    def _autologin_record_likely_username(self, record: dict[str, Any]) -> bool:
        if bool(record.get("is_password", False)):
            return False
        fp = str(record.get("fingerprint", "")).lower()
        if any(token in fp for token in ("search", "coupon", "promo", "captcha", "otp", "2fa", "verification")):
            return False
        if any(token in fp for token in ("user", "email", "login", "account", "identifier", "phone")):
            return True
        input_type = str(record.get("input_type", "")).lower()
        return input_type in {"", "text", "email", "tel"}

    def _autologin_credentials_from_records(self, site_key: str) -> tuple[str, str]:
        records = self._load_login_form_memory_snapshot()
        username = ""
        password = ""
        username_updated_at = -1
        password_updated_at = -1

        for record in records:
            if not isinstance(record, dict):
                continue
            origin = str(record.get("origin", "")).strip()
            host = self._normalize_site_key(origin)
            if not host or not self._autologin_site_matches(site_key, host):
                continue
            value = str(record.get("value", ""))
            if not value.strip():
                continue
            updated_at = _to_non_negative_int(record.get("updated_at", 0))

            if bool(record.get("is_password", False)):
                if updated_at >= password_updated_at:
                    password = value
                    password_updated_at = updated_at
                continue

            if not self._autologin_record_likely_username(record):
                continue
            if updated_at >= username_updated_at:
                username = value.strip()
                username_updated_at = updated_at

        return username, password

    def _autologin_credentials(self, site_key: str) -> tuple[str, str]:
        credentials = self._load_login_credentials_snapshot()
        entry: dict[str, Any] = {}
        if isinstance(credentials, dict):
            direct = credentials.get(site_key)
            if isinstance(direct, dict):
                entry = direct
            else:
                for raw_key, raw_value in credentials.items():
                    key = self._normalize_site_key(str(raw_key))
                    if not key or not isinstance(raw_value, dict):
                        continue
                    if site_key == key or site_key.endswith(f".{key}") or key.endswith(f".{site_key}"):
                        entry = raw_value
                        break
        username = str(entry.get("username", "")).strip()
        password = str(entry.get("password", ""))
        if username and password:
            return username, password

        fallback_username, fallback_password = self._autologin_credentials_from_records(site_key)
        if not username:
            username = fallback_username
        if not password:
            password = fallback_password
        return username, password

    def _autologin_entry(self, site_key: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with _AUTOLOGIN_STATS_LOCK:
            payload = _load_autologin_stats()
            sites = payload.setdefault("sites", {})
            raw = sites.get(site_key)
            if not isinstance(raw, dict):
                raw = {}

            entry = {
                "attempts_since_successful_login": _to_non_negative_int(raw.get("attempts_since_successful_login", 0)),
                "consecutive_autologin_failures": _to_non_negative_int(raw.get("consecutive_autologin_failures", 0)),
                "backoff_remaining": _to_non_negative_int(raw.get("backoff_remaining", 0)),
                "last_result": str(raw.get("last_result", "")),
                "updated_at": str(raw.get("updated_at", now)),
            }
            sites[site_key] = entry
            _save_autologin_stats(payload)
            return dict(entry)

    def _autologin_update(self, site_key: str, updater: Any) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with _AUTOLOGIN_STATS_LOCK:
            payload = _load_autologin_stats()
            sites = payload.setdefault("sites", {})
            raw = sites.get(site_key)
            if not isinstance(raw, dict):
                raw = {}

            entry = {
                "attempts_since_successful_login": _to_non_negative_int(raw.get("attempts_since_successful_login", 0)),
                "consecutive_autologin_failures": _to_non_negative_int(raw.get("consecutive_autologin_failures", 0)),
                "backoff_remaining": _to_non_negative_int(raw.get("backoff_remaining", 0)),
                "last_result": str(raw.get("last_result", "")),
                "updated_at": str(raw.get("updated_at", now)),
            }

            updater(entry)
            entry["attempts_since_successful_login"] = _to_non_negative_int(
                entry.get("attempts_since_successful_login", 0)
            )
            entry["consecutive_autologin_failures"] = _to_non_negative_int(
                entry.get("consecutive_autologin_failures", 0)
            )
            entry["backoff_remaining"] = _to_non_negative_int(entry.get("backoff_remaining", 0))
            entry["last_result"] = str(entry.get("last_result", ""))
            entry["updated_at"] = now

            sites[site_key] = entry
            _save_autologin_stats(payload)
            return dict(entry)

    def _autologin_mark_success(self, site_key: str) -> None:
        def _apply(entry: dict[str, Any]) -> None:
            entry["attempts_since_successful_login"] = 0
            entry["consecutive_autologin_failures"] = 0
            entry["backoff_remaining"] = 0
            entry["last_result"] = "autologin_success"

        self._autologin_update(site_key, _apply)

    def _autologin_mark_failure(self, site_key: str) -> None:
        def _apply(entry: dict[str, Any]) -> None:
            failures = _to_non_negative_int(entry.get("consecutive_autologin_failures", 0)) + 1
            entry["attempts_since_successful_login"] = _to_non_negative_int(
                entry.get("attempts_since_successful_login", 0)
            ) + 1
            entry["consecutive_autologin_failures"] = failures
            entry["backoff_remaining"] = (2 ** (failures - 1)) if AUTOLOGIN_BACKOFF_ENABLED else 0
            entry["last_result"] = "autologin_failure"

        self._autologin_update(site_key, _apply)

    def _autologin_mark_skipped_backoff(self, site_key: str) -> None:
        def _apply(entry: dict[str, Any]) -> None:
            remaining = _to_non_negative_int(entry.get("backoff_remaining", 0))
            if remaining > 0:
                remaining -= 1
            entry["attempts_since_successful_login"] = _to_non_negative_int(
                entry.get("attempts_since_successful_login", 0)
            ) + 1
            entry["backoff_remaining"] = remaining
            entry["last_result"] = "skipped_backoff"

        self._autologin_update(site_key, _apply)

    def _autologin_mark_skipped_no_credentials(self, site_key: str) -> None:
        def _apply(entry: dict[str, Any]) -> None:
            entry["attempts_since_successful_login"] = _to_non_negative_int(
                entry.get("attempts_since_successful_login", 0)
            ) + 1
            entry["last_result"] = "skipped_no_credentials"

        self._autologin_update(site_key, _apply)

    async def _attempt_autologin_if_eligible(self) -> bool:
        site_key = self._autologin_site_key()
        if not site_key:
            return False

        started_at = datetime.now(timezone.utc).isoformat()
        state = self._autologin_entry(site_key)
        username, password = self._autologin_credentials(site_key)
        if not username or not password:
            self._autologin_mark_skipped_no_credentials(site_key)
            state_after = self._autologin_entry(site_key)
            self._append_autologin_attempt_record(
                {
                    "timestamp": started_at,
                    "site": site_key,
                    "target_url": self.target_url,
                    "attempted": False,
                    "status": "skipped_no_credentials",
                    "credentials_available": False,
                    "backoff_remaining_before": _to_non_negative_int(state.get("backoff_remaining", 0)),
                    "stats_before": state,
                    "stats_after": state_after,
                }
            )
            print(f"[autologin] skipped for {site_key}: missing username/password credentials", flush=True)
            return False

        if AUTOLOGIN_BACKOFF_ENABLED and _to_non_negative_int(state.get("backoff_remaining", 0)) > 0:
            self._autologin_mark_skipped_backoff(site_key)
            state_after = self._autologin_entry(site_key)
            remaining = _to_non_negative_int(state_after.get("backoff_remaining", 0))
            self._append_autologin_attempt_record(
                {
                    "timestamp": started_at,
                    "site": site_key,
                    "target_url": self.target_url,
                    "attempted": False,
                    "status": "skipped_backoff",
                    "credentials_available": True,
                    "backoff_remaining_before": _to_non_negative_int(state.get("backoff_remaining", 0)),
                    "backoff_remaining_after": remaining,
                    "stats_before": state,
                    "stats_after": state_after,
                }
            )
            print(
                f"[autologin] skipped for {site_key}: backoff active (remaining_skips={remaining})",
                flush=True,
            )
            return False

        print(f"[autologin] attempting for {site_key} (headless)", flush=True)
        attempt_ok = False
        run_record: dict[str, Any] | None = None
        autologin_artifact: dict[str, Any] | None = None
        autologin_artifact_error: str | None = None
        try:
            try:
                captured = await self.recorder.capture(
                    self.page,
                    source="autologin_before",
                    metadata={"site": site_key, "target_url": self.target_url},
                )
                autologin_artifact = captured["event"]
            except Exception as capture_exc:  # noqa: BLE001
                autologin_artifact_error = str(capture_exc)

            await self._install_login_form_prefill_init_script()
            await _safe_goto(self.page, self.target_url)
            run_record = await self._run_autologin_reveal_flow(site_key=site_key)
            await self.page.wait_for_timeout(2500)
            try:
                await self.page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            is_logged_in = await self._is_logged_in_page(self.page, self.target_url)
            if not is_logged_in:
                # Re-check from target URL in case submit redirected away temporarily.
                await _safe_goto(self.page, self.target_url)
                is_logged_in = await self._is_logged_in_page(self.page, self.target_url)

            attempt_ok = bool(is_logged_in)
        except Exception as exc:  # noqa: BLE001
            self._write_llm_call_log(
                "autologin_error",
                {"site": site_key, "target_url": self.target_url},
                None,
                error=exc,
            )
            attempt_ok = False

        final_capture_event: dict[str, Any] | None = None
        final_capture_error: str | None = None
        try:
            final_capture = await self.recorder.capture(
                self.page,
                source="autologin_result",
                metadata={"site": site_key, "target_url": self.target_url, "success": attempt_ok},
            )
            final_capture_event = final_capture["event"]
        except Exception as exc:  # noqa: BLE001
            final_capture_error = str(exc)

        if attempt_ok:
            try:
                await self._persist_session_storage_from_context()
            except Exception:
                pass
            try:
                await self._persist_cookies()
            except Exception:
                pass
            self._autologin_mark_success(site_key)
            state_after = self._autologin_entry(site_key)
            self._append_autologin_attempt_record(
                {
                    "timestamp": started_at,
                    "site": site_key,
                    "target_url": self.target_url,
                    "attempted": True,
                    "status": "autologin_success",
                    "credentials_available": True,
                    "backoff_remaining_before": _to_non_negative_int(state.get("backoff_remaining", 0)),
                    "stats_before": state,
                    "stats_after": state_after,
                    "before_artifact": autologin_artifact,
                    "before_artifact_error": autologin_artifact_error,
                    "result_artifact": final_capture_event,
                    "result_artifact_error": final_capture_error,
                    "steps_run_timestamp": run_record.get("timestamp") if isinstance(run_record, dict) else None,
                }
            )
            print(f"[autologin] success for {site_key}", flush=True)
            return True

        self._autologin_mark_failure(site_key)
        state_after = self._autologin_entry(site_key)
        self._append_autologin_attempt_record(
            {
                "timestamp": started_at,
                "site": site_key,
                "target_url": self.target_url,
                "attempted": True,
                "status": "autologin_failure",
                "credentials_available": True,
                "backoff_remaining_before": _to_non_negative_int(state.get("backoff_remaining", 0)),
                "backoff_remaining_after": _to_non_negative_int(state_after.get("backoff_remaining", 0)),
                "stats_before": state,
                "stats_after": state_after,
                "before_artifact": autologin_artifact,
                "before_artifact_error": autologin_artifact_error,
                "result_artifact": final_capture_event,
                "result_artifact_error": final_capture_error,
                "steps_run_timestamp": run_record.get("timestamp") if isinstance(run_record, dict) else None,
            }
        )
        print(
            "[autologin] failed for "
            f"{site_key} (next_backoff_skips={_to_non_negative_int(state_after.get('backoff_remaining', 0))})",
            flush=True,
        )
        return False

    async def _run_autologin_reveal_flow(self, *, site_key: str) -> dict[str, Any] | None:
        instruction = (
            "Log the user into this website. "
            "Username/email and password will be autofilled automatically by the page script, so do NOT type credentials. "
            "Your job is to navigate login UI and click the needed buttons (sign in, continue, submit, next, etc.) "
            "so login completes successfully. "
            "You may click, scroll, and wait as needed. "
            "Stop only after login appears complete."
        )

        initial = await self.recorder.capture(
            self.page,
            source="autologin_start",
            metadata={"site": site_key, "target_url": self.target_url},
        )
        initial_b64 = base64.b64encode(initial["screenshot_bytes"]).decode("ascii")
        latest_view = initial

        tools = [
            {
                "type": "computer_use_preview",
                "display_width": DISPLAY_WIDTH,
                "display_height": DISPLAY_HEIGHT,
                "environment": "browser",
            }
        ]

        log_idx = self._next_llm_index("autologin_loop")
        initial_request = {
            "model": COMPUTER_USE_MODEL,
            "tools": tools,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{initial_b64}"},
                    ],
                }
            ],
            "truncation": "auto",
        }

        try:
            response = self.openai_client.responses.create(**initial_request)
            self._write_llm_call_log("autologin_loop", initial_request, response, index=log_idx)
        except Exception as exc:  # noqa: BLE001
            self._write_llm_call_log("autologin_loop", initial_request, None, index=log_idx, error=exc)
            return None

        steps_out: list[dict[str, Any]] = []
        final_text = ""

        for step_idx in range(AUTOLOGIN_MAX_ACTION_STEPS):
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
            computer_action = action_obj.model_dump() if hasattr(action_obj, "model_dump") else dict(action_obj)
            action_type = str(computer_action.get("type", "")).strip()
            if action_type == "screenshot":
                captured = latest_view
                screenshot_b64 = base64.b64encode(captured["screenshot_bytes"]).decode("ascii")
                step_entry = {
                    "step": step_idx,
                    "computer_action": computer_action,
                    "artifact": captured["event"],
                    "artifact_reused": True,
                }
            else:
                plan_command = _computer_action_to_plan_command(computer_action)
                action_error = None
                try:
                    await _execute_plan_interaction(self.page, plan_command, allow_text_entry=False)
                except Exception as exc:  # noqa: BLE001
                    action_error = str(exc)

                captured = await self.recorder.capture(
                    self.page,
                    source="autologin_action",
                    metadata={
                        "site": site_key,
                        "step": step_idx,
                        "plan_command": _plan_command_payload(plan_command),
                        "error": action_error,
                    },
                )
                latest_view = captured
                screenshot_b64 = base64.b64encode(captured["screenshot_bytes"]).decode("ascii")
                step_entry = {
                    "step": step_idx,
                    "plan_command": _plan_command_payload(plan_command),
                    "error": action_error,
                    "artifact": captured["event"],
                }
                if action_error:
                    print(
                        f"[autologin step {step_idx + 1}/{AUTOLOGIN_MAX_ACTION_STEPS}] error: {action_error}",
                        flush=True,
                    )
                else:
                    print(
                        f"[autologin step {step_idx + 1}/{AUTOLOGIN_MAX_ACTION_STEPS}] "
                        f"{_plan_command_line(plan_command)}",
                        flush=True,
                    )
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
                    "autologin_loop",
                    followup_request,
                    response,
                    index=self._next_llm_index("autologin_loop"),
                )
            except Exception as exc:  # noqa: BLE001
                self._write_llm_call_log(
                    "autologin_loop",
                    followup_request,
                    None,
                    index=self._next_llm_index("autologin_loop"),
                    error=exc,
                )
                break

        run_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "site": site_key,
            "target_url": self.target_url,
            "instruction": instruction,
            "max_command_steps": AUTOLOGIN_MAX_ACTION_STEPS,
            "final_text": final_text,
            "steps": steps_out,
        }
        runs_path = self.artifact_dir / "autologin_steps.json"
        payload: dict[str, Any] = {"runs": []}
        if runs_path.exists():
            try:
                existing = json.loads(runs_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                    payload = existing
            except Exception:
                payload = {"runs": []}
        payload.setdefault("runs", [])
        if isinstance(payload["runs"], list):
            payload["runs"].append(run_record)
        else:
            payload["runs"] = [run_record]
        runs_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return run_record


# stats helpers

def _to_non_negative_int(value: Any) -> int:
    try:
        out = int(value)
    except Exception:
        return 0
    return out if out >= 0 else 0


def _load_autologin_stats() -> dict[str, Any]:
    if not AUTOLOGIN_STATS_FILE.exists():
        return {"version": 1, "sites": {}}

    try:
        payload = json.loads(AUTOLOGIN_STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "sites": {}}

    if not isinstance(payload, dict):
        return {"version": 1, "sites": {}}

    sites = payload.get("sites")
    if not isinstance(sites, dict):
        payload["sites"] = {}
    return payload


def _save_autologin_stats(payload: dict[str, Any]) -> None:
    out = dict(payload)
    out["version"] = 1
    out["saved_at"] = datetime.now(timezone.utc).isoformat()
    sites = out.get("sites")
    if not isinstance(sites, dict):
        out["sites"] = {}

    AUTOLOGIN_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTOLOGIN_STATS_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")

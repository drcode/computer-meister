from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from execution_common import ANSWER_MODEL, ManualLoginRequiredError, _responses_text, _safe_goto


class LoginMixin:
    async def _ensure_login(self) -> None:
        await _safe_goto(self.page, self.target_url)
        is_logged_in = await self._is_logged_in_page(self.page, self.target_url)
        if is_logged_in:
            return

        if not self.allow_manual_login:
            raise ManualLoginRequiredError(
                f"Manual login required for {self.query.section_id}; skipped in scrape-headless mode."
            )

        await self._close_context()

        self.login_prompt_lock.acquire()
        try:
            self.headless = False
            self.help_used = True
            await self._launch_context(headless=False)
            prefill_enabled = not self.query.nofill
            capture_enabled = prefill_enabled and not self.query.nocapture
            if prefill_enabled:
                await self._install_login_form_prefill_init_script()
            if capture_enabled:
                await self._install_login_field_capture()
            await _safe_goto(self.page, self.target_url)
            print(
                f"\nLogin required for {self.query.section_id}. "
                "Please complete login in the opened browser, then press ENTER here to continue.",
                flush=True,
            )
            await asyncio.get_event_loop().run_in_executor(None, input)
            await self.page.wait_for_timeout(300)
            if capture_enabled:
                self._persist_login_form_memory()
            await self._persist_session_storage_from_context()
            await self._persist_cookies()
        finally:
            await self._close_context()
            self.login_prompt_lock.release()

        await self._launch_context(headless=True)
        await _safe_goto(self.page, self.target_url)

    async def _is_logged_in_page(self, page: Any, url: str) -> bool:
        # Give dynamic sites a short window to render before we classify login state.
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        screenshot = b""
        for _ in range(3):
            await page.wait_for_timeout(700)
            screenshot = await page.screenshot()
            html = await page.content()
            # Heuristic to avoid classifying obviously blank/early frames.
            if len(screenshot) >= 5000 and len(html.strip()) >= 300:
                break

        log_index = self._next_llm_index("is_logged_in_check")
        input_image_name = f"is_logged_in_check_input_{log_index}.png"
        input_image_path = self.artifact_dir / input_image_name
        input_image_path.write_bytes(screenshot)
        screenshot_b64 = base64.b64encode(screenshot).decode("ascii")
        prompt = (
            "Is the user already logged in on this webpage? "
            "Look for account/avatar/sign out indicators. "
            "Respond with exactly YES or NO."
        )
        api_request = {
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
            "max_output_tokens": 16,
        }
        log_request = dict(api_request)
        log_request["input_image_artifact"] = input_image_name

        try:
            response = self.openai_client.responses.create(**api_request)
            self._write_llm_call_log("is_logged_in_check", log_request, response, index=log_index)
        except Exception as exc:  # noqa: BLE001
            self._write_llm_call_log("is_logged_in_check", log_request, None, index=log_index, error=exc)
            return False

        text = _responses_text(response).strip().lower()
        return text.startswith("yes")

    def _load_session_storage_snapshot(self) -> dict[str, dict[str, str]]:
        if not self.session_storage_path.exists():
            return {}
        try:
            payload = json.loads(self.session_storage_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        cleaned: dict[str, dict[str, str]] = {}
        for origin, items in payload.items():
            if not isinstance(origin, str) or not isinstance(items, dict):
                continue
            normalized: dict[str, str] = {}
            for key, value in items.items():
                if not isinstance(key, str):
                    continue
                if value is None:
                    normalized[key] = ""
                elif isinstance(value, str):
                    normalized[key] = value
                else:
                    normalized[key] = str(value)
            if normalized:
                cleaned[origin] = normalized
        return cleaned

    def _load_login_form_memory_snapshot(self) -> list[dict[str, Any]]:
        if not self.login_form_memory_path.exists():
            return []
        try:
            payload = json.loads(self.login_form_memory_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            return []

        out: list[dict[str, Any]] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            origin = item.get("origin")
            path = item.get("path")
            fingerprint = item.get("fingerprint")
            value = item.get("value")
            if not isinstance(origin, str) or not origin:
                continue
            if not isinstance(path, str):
                path = "/"
            if not isinstance(fingerprint, str) or not fingerprint:
                continue
            if value is None:
                value = ""
            if not isinstance(value, str):
                value = str(value)

            updated_at = item.get("updated_at")
            if not isinstance(updated_at, int):
                updated_at = int(time.time() * 1000)

            out.append(
                {
                    "origin": origin,
                    "path": path,
                    "fingerprint": fingerprint,
                    "value": value,
                    "is_password": bool(item.get("is_password", False)),
                    "input_type": str(item.get("input_type", "")),
                    "updated_at": updated_at,
                }
            )
        return out

    def _normalize_site_key(self, value: str) -> str:
        raw = (value or "").strip().lower()
        if not raw:
            return ""
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        host = (parsed.hostname or raw).strip().lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _load_login_credentials_snapshot(self) -> dict[str, dict[str, Any]]:
        if not self.login_form_memory_path.exists():
            return {}
        try:
            payload = json.loads(self.login_form_memory_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw = payload.get("credentials") if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            return {}

        out: dict[str, dict[str, Any]] = {}
        now_ms = int(time.time() * 1000)
        for site_key, item in raw.items():
            site = self._normalize_site_key(str(site_key))
            if not site or not isinstance(item, dict):
                continue
            username = item.get("username", "")
            password = item.get("password", "")
            if username is None:
                username = ""
            if password is None:
                password = ""
            if not isinstance(username, str):
                username = str(username)
            if not isinstance(password, str):
                password = str(password)
            updated_at = item.get("updated_at")
            if not isinstance(updated_at, int):
                updated_at = now_ms
            out[site] = {
                "username": username,
                "password": password,
                "updated_at": updated_at,
            }
        return out

    def _save_login_form_memory_snapshot(self, records: list[dict[str, Any]]) -> None:
        self.login_form_memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if self.login_form_memory_path.exists():
            try:
                existing = json.loads(self.login_form_memory_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = dict(existing)
            except Exception:
                payload = {}
        version = payload.get("version", 1)
        try:
            payload["version"] = int(version)
        except Exception:
            payload["version"] = 1
        payload["saved_at"] = datetime.now(timezone.utc).isoformat()
        payload["records"] = records
        self.login_form_memory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async def _install_login_form_prefill_init_script(self) -> None:
        if self._context is None:
            return
        records = self._load_login_form_memory_snapshot()
        credentials = self._load_login_credentials_snapshot()
        if not records and not credentials:
            return
        serialized = json.dumps(records)
        credentials_serialized = json.dumps(credentials)
        script = f"""
(() => {{
  const records = {serialized};
  const credentials = {credentials_serialized};
  const excludedTypes = new Set(["hidden", "submit", "button", "reset", "file", "image", "checkbox", "radio"]);

  const norm = (value) => (value || "").toString().trim().toLowerCase();
  const safeText = (value) => (value || "").toString().trim().replace(/\\s+/g, " ").slice(0, 120).toLowerCase();
  const textFor = (node) => safeText((node && (node.innerText || node.textContent)) || "");

  const fieldFingerprint = (field) => {{
    const tag = norm(field.tagName);
    const type = norm(field.type);
    const id = norm(field.id);
    const name = norm(field.getAttribute("name"));
    const autocomplete = norm(field.getAttribute("autocomplete"));
    let placeholder = safeText(field.getAttribute("placeholder"));
    if (!placeholder) {{
      placeholder = safeText(field.placeholder);
    }}
    let label = "";
    if (id) {{
      const forLabel = document.querySelector(`label[for="${{CSS.escape(id)}}"]`);
      if (forLabel) {{
        label = textFor(forLabel);
      }}
    }}
    if (!label) {{
      const parentLabel = field.closest("label");
      if (parentLabel) {{
        label = textFor(parentLabel);
      }}
    }}
    if (!label) {{
      const labelledBy = field.getAttribute("aria-labelledby");
      if (labelledBy) {{
        for (const token of labelledBy.split(/\\s+/g)) {{
          const el = document.getElementById(token);
          if (!el) continue;
          label += " " + textFor(el);
        }}
        label = safeText(label);
      }}
    }}
    return [tag, type, autocomplete, name, id, placeholder, label].join("|");
  }};

  const normalizeHost = (value) => {{
    let host = norm(value);
    if (!host) return "";
    if (host.includes("://")) {{
      try {{
        host = norm(new URL(host).hostname || "");
      }} catch (_) {{}}
    }}
    if (host.startsWith("www.")) {{
      host = host.slice(4);
    }}
    return host;
  }};

  const normalizedCredentials = {{}};
  for (const [site, item] of Object.entries(credentials || {{}})) {{
    if (!item || typeof item !== "object") continue;
    const host = normalizeHost(site);
    if (!host) continue;
    const username = typeof item.username === "string" ? item.username : "";
    const password = typeof item.password === "string" ? item.password : "";
    normalizedCredentials[host] = {{ username, password }};
  }}

  const fieldHints = (field) => {{
    const type = norm(field.type);
    const id = norm(field.id);
    const name = norm(field.getAttribute("name"));
    const autocomplete = norm(field.getAttribute("autocomplete"));
    let placeholder = safeText(field.getAttribute("placeholder"));
    if (!placeholder) {{
      placeholder = safeText(field.placeholder);
    }}
    let label = "";
    if (id) {{
      const forLabel = document.querySelector(`label[for="${{CSS.escape(id)}}"]`);
      if (forLabel) {{
        label = textFor(forLabel);
      }}
    }}
    if (!label) {{
      const parentLabel = field.closest("label");
      if (parentLabel) {{
        label = textFor(parentLabel);
      }}
    }}
    if (!label) {{
      const labelledBy = field.getAttribute("aria-labelledby");
      if (labelledBy) {{
        for (const token of labelledBy.split(/\\s+/g)) {{
          const el = document.getElementById(token);
          if (!el) continue;
          label += " " + textFor(el);
        }}
        label = safeText(label);
      }}
    }}
    return {{ type, id, name, autocomplete, placeholder, label }};
  }};

  const exactMap = new Map();
  const originMap = new Map();
  for (const item of records) {{
    if (!item || typeof item !== "object") continue;
    const origin = item.origin || "";
    const path = item.path || "/";
    const fp = item.fingerprint || "";
    const value = item.value;
    if (!origin || !fp || typeof value !== "string") continue;
    const updated = Number(item.updated_at || 0);
    const exactKey = `${{origin}}|${{path}}|${{fp}}`;
    const prevExact = exactMap.get(exactKey);
    if (!prevExact || Number(prevExact.updated_at || 0) <= updated) {{
      exactMap.set(exactKey, item);
    }}
    const originKey = `${{origin}}|${{fp}}`;
    const prevOrigin = originMap.get(originKey);
    if (!prevOrigin || Number(prevOrigin.updated_at || 0) <= updated) {{
      originMap.set(originKey, item);
    }}
  }}

	  const prefillState = new WeakMap();

	  const markAutofillEvent = (event) => {{
	    if (!event) return event;
	    try {{
	      Object.defineProperty(event, "__cmAutofill", {{ value: true }});
	    }} catch (_) {{
	      try {{
	        event.__cmAutofill = true;
	      }} catch (_) {{}}
	    }}
	    return event;
	  }};

	  const makeEvent = (type) => markAutofillEvent(new Event(type, {{ bubbles: true, composed: true, cancelable: true }}));
	  const makeInputEvent = (type) => {{
	    try {{
	      return markAutofillEvent(
	        new InputEvent(type, {{
	          bubbles: true,
	          composed: true,
	          cancelable: true,
	          inputType: "insertReplacementText",
	          data: null,
	        }})
	      );
	    }} catch (_) {{
	      return makeEvent(type);
	    }}
	  }};

	  const dispatch = (field, event) => {{
	    try {{
	      field.dispatchEvent(event);
	    }} catch (_) {{}}
	  }};

	  const dispatchValueEvents = (field) => {{
	    dispatch(field, makeInputEvent("beforeinput"));
	    dispatch(field, makeInputEvent("input"));
	    dispatch(field, makeEvent("change"));
	  }};

	  // Use the native "value" setter to play nicely with React/value trackers.
	  const setNativeValue = (element, value) => {{
	    try {{
	      const valueDescriptor = Object.getOwnPropertyDescriptor(element, "value");
	      const valueSetter = valueDescriptor && valueDescriptor.set;
	      const prototype = Object.getPrototypeOf(element);
	      const prototypeDescriptor = Object.getOwnPropertyDescriptor(prototype, "value");
	      const prototypeValueSetter = prototypeDescriptor && prototypeDescriptor.set;
	      if (prototypeValueSetter && valueSetter !== prototypeValueSetter) {{
	        prototypeValueSetter.call(element, value);
	      }} else if (valueSetter) {{
	        valueSetter.call(element, value);
	      }} else {{
	        element.value = value;
	      }}
	    }} catch (_) {{
	      try {{
	        element.value = value;
	      }} catch (_) {{}}
	    }}
	  }};

	  const eligibleField = (field) => {{
	    if (!(field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement)) return false;
	    const type = norm(field.type);
	    if (excludedTypes.has(type)) return false;
	    if (field.disabled || field.readOnly) return false;
	    return true;
	  }};

  const isUsernameCandidate = (hints) => {{
    if (!hints || typeof hints !== "object") return false;
    if (hints.type === "password") return false;
    if (["hidden", "search", "url", "number", "date", "datetime-local", "time"].includes(hints.type)) {{
      return false;
    }}
    const combined = [hints.name, hints.id, hints.autocomplete, hints.placeholder, hints.label].join(" ");
    if (/search|coupon|promo|verification|otp|one-time|2fa|captcha/i.test(combined)) {{
      return false;
    }}
    if (/user|email|e-mail|login|account|identifier|phone/.test(combined)) {{
      return true;
    }}
    return hints.type === "email";
  }};

  const credentialValueForField = (field, origin) => {{
    let host = "";
    try {{
      host = normalizeHost(new URL(origin).hostname || "");
    }} catch (_) {{
      host = normalizeHost(window.location.hostname || "");
    }}
    const creds = normalizedCredentials[host];
    if (!creds) return null;
    const hints = fieldHints(field);
    if (hints.type === "password") {{
      if (typeof creds.password === "string" && creds.password.length > 0) {{
        return creds.password;
      }}
      return null;
    }}
    if (!isUsernameCandidate(hints)) return null;
    if (typeof creds.username === "string" && creds.username.length > 0) {{
      return creds.username;
    }}
    return null;
  }};

  const matchForField = (field, origin, path) => {{
    const fp = fieldFingerprint(field);
    if (fp) {{
      const exactKey = `${{origin}}|${{path}}|${{fp}}`;
      const originKey = `${{origin}}|${{fp}}`;
      const match = exactMap.get(exactKey) || originMap.get(originKey);
      if (match && typeof match.value === "string") {{
        return {{ fp, value: match.value }};
      }}
    }}
    const fallback = credentialValueForField(field, origin);
    if (typeof fallback === "string" && fallback.length > 0) {{
      return {{ fp: fp || "__site_credentials__", value: fallback }};
    }}
    return null;
  }};

	  const applyField = (field, origin, path, now) => {{
	    if (!eligibleField(field)) return;
	    const match = matchForField(field, origin, path);
	    if (!match) return;

	    const currentValue = (field.value || "").toString();
	    const state = prefillState.get(field);

	    if (!currentValue) {{
	      setNativeValue(field, match.value);
	      dispatchValueEvents(field);
	      prefillState.set(field, {{ value: match.value, filledAt: now, rearmed: false }});
	      return;
	    }}

	    // Some sites/frameworks only attach input handlers after hydration. If we filled before that,
	    // replay a single "change" in a way that triggers value watchers (React) without user input.
	    if (!state) return;
	    if (state.rearmed) return;
	    if (state.value !== match.value) return;
	    if (currentValue !== match.value) return;
	    if (document.activeElement === field) return;

	    const ageMs = now - Number(state.filledAt || 0);
	    if (ageMs < 150) return;
	    if (ageMs > 8000) {{
	      state.rearmed = true;
	      return;
	    }}

	    const tracker = field && field._valueTracker;
	    if (tracker && typeof tracker.setValue === "function") {{
	      try {{
	        tracker.setValue("");
	      }} catch (_) {{}}
	      dispatch(field, makeInputEvent("input"));
	      dispatch(field, makeEvent("change"));
	      state.rearmed = true;
	      return;
	    }}

	    let hasPatchedValueSetter = false;
	    try {{
	      const valueDescriptor = Object.getOwnPropertyDescriptor(field, "value");
	      const prototype = Object.getPrototypeOf(field);
	      const prototypeDescriptor = Object.getOwnPropertyDescriptor(prototype, "value");
	      hasPatchedValueSetter = Boolean(
	        valueDescriptor &&
	          prototypeDescriptor &&
	          typeof valueDescriptor.set === "function" &&
	          typeof prototypeDescriptor.set === "function" &&
	          valueDescriptor.set !== prototypeDescriptor.set
	      );
	    }} catch (_) {{}}

	    if (!hasPatchedValueSetter) {{
	      // Late-bound listeners: re-dispatch without mutating the visible value.
	      dispatchValueEvents(field);
	      state.rearmed = true;
	      return;
	    }}

	    // Patched setters (React-like): clear + restore to force a detected change.
	    setNativeValue(field, "");
	    dispatchValueEvents(field);
	    setNativeValue(field, match.value);
	    dispatchValueEvents(field);
	    state.rearmed = true;
	  }};

	  const applyPrefill = () => {{
	    const origin = window.location.origin;
	    const path = window.location.pathname || "/";
	    const now = Date.now();
	    const fields = document.querySelectorAll("input, textarea");
	    for (const field of fields) {{
	      if (!(field instanceof HTMLElement)) continue;
	      applyField(field, origin, path, now);
	    }}
	  }};

	  const scheduleApply = () => {{
	    applyPrefill();
	    setTimeout(applyPrefill, 250);
	    setTimeout(applyPrefill, 1000);
	    setTimeout(applyPrefill, 2500);
	    setTimeout(applyPrefill, 6000);
	  }};

	  if (document.readyState === "loading") {{
	    document.addEventListener("DOMContentLoaded", scheduleApply, {{ once: true }});
	  }} else {{
	    scheduleApply();
	  }}

	  let queued = false;
	  const queueApply = () => {{
	    if (queued) return;
	    queued = true;
	    setTimeout(() => {{
	      queued = false;
	      applyPrefill();
	    }}, 150);
	  }};

	  try {{
	    const observer = new MutationObserver(queueApply);
	    observer.observe(document.documentElement, {{ childList: true, subtree: true }});
	  }} catch (_) {{}}

	  document.addEventListener(
	    "focusin",
	    (event) => {{
	      const field = event && event.target;
	      if (!(field instanceof HTMLElement)) return;
	      const origin = window.location.origin;
	      const path = window.location.pathname || "/";
	      const now = Date.now();
	      applyField(field, origin, path, now);
	    }},
	    true
	  );
	}})();
	"""
        await self._context.add_init_script(script=script)

    async def _install_human_login_stealth_init_script(self) -> None:
        if self._context is None:
            return
        script = """
(() => {
  try {
    Object.defineProperty(navigator, "webdriver", {
      get: () => undefined,
      configurable: true,
    });
  } catch (_) {}
  try {
    delete window.__playwright__binding__;
  } catch (_) {}
  try {
    delete window.__pwInitScripts;
  } catch (_) {}
})();
"""
        await self._context.add_init_script(script=script)

    async def _install_login_field_capture(self) -> None:
        if self._context is None:
            return
        self._captured_login_fields = {}

        def _capture_binding(_: Any, payload: Any) -> None:
            if not isinstance(payload, dict):
                return
            origin = payload.get("origin")
            path = payload.get("path")
            fingerprint = payload.get("fingerprint")
            value = payload.get("value")
            if not isinstance(origin, str) or not origin:
                return
            if not isinstance(path, str):
                path = "/"
            if not isinstance(fingerprint, str) or not fingerprint:
                return
            if value is None:
                value = ""
            if not isinstance(value, str):
                value = str(value)
            record = {
                "origin": origin,
                "path": path,
                "fingerprint": fingerprint,
                "value": value,
                "is_password": bool(payload.get("is_password", False)),
                "input_type": str(payload.get("input_type", "")),
                "updated_at": int(payload.get("captured_at_ms", int(time.time() * 1000))),
            }
            key = (record["origin"], record["path"], record["fingerprint"])
            prior = self._captured_login_fields.get(key)
            if prior is None or int(prior.get("updated_at", 0)) <= int(record["updated_at"]):
                self._captured_login_fields[key] = record

        await self._context.expose_binding("cmRecordLoginField", _capture_binding)

        script = """
(() => {
  if (window.__cmLoginCaptureInstalled) return;
  window.__cmLoginCaptureInstalled = true;

  const excludedTypes = new Set(["hidden", "submit", "button", "reset", "file", "image", "checkbox", "radio"]);
  const norm = (value) => (value || "").toString().trim().toLowerCase();
  const safeText = (value) => (value || "").toString().trim().replace(/\\s+/g, " ").slice(0, 120).toLowerCase();
  const textFor = (node) => safeText((node && (node.innerText || node.textContent)) || "");

  const fieldFingerprint = (field) => {
    const tag = norm(field.tagName);
    const type = norm(field.type);
    const id = norm(field.id);
    const name = norm(field.getAttribute("name"));
    const autocomplete = norm(field.getAttribute("autocomplete"));
    let placeholder = safeText(field.getAttribute("placeholder"));
    if (!placeholder) {
      placeholder = safeText(field.placeholder);
    }
    let label = "";
    if (id) {
      const forLabel = document.querySelector(`label[for="${CSS.escape(id)}"]`);
      if (forLabel) {
        label = textFor(forLabel);
      }
    }
    if (!label) {
      const parentLabel = field.closest("label");
      if (parentLabel) {
        label = textFor(parentLabel);
      }
    }
    if (!label) {
      const labelledBy = field.getAttribute("aria-labelledby");
      if (labelledBy) {
        for (const token of labelledBy.split(/\\s+/g)) {
          const el = document.getElementById(token);
          if (!el) continue;
          label += " " + textFor(el);
        }
        label = safeText(label);
      }
    }
    return [tag, type, autocomplete, name, id, placeholder, label].join("|");
  };

  const eligible = (field) => {
    if (!field) return false;
    if (!(field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement)) return false;
    const type = norm(field.type);
    if (excludedTypes.has(type)) return false;
    if (field.disabled || field.readOnly) return false;
    return true;
  };

  const captureField = (field) => {
    if (!eligible(field)) return;
    const payload = {
      origin: window.location.origin || "",
      path: window.location.pathname || "/",
      fingerprint: fieldFingerprint(field),
      value: (field.value || "").toString(),
      input_type: norm(field.type),
      is_password: norm(field.type) === "password",
      captured_at_ms: Date.now(),
    };
    if (!payload.origin || !payload.fingerprint) return;
    if (typeof window.cmRecordLoginField === "function") {
      window.cmRecordLoginField(payload);
    }
  };

	  document.addEventListener("change", (event) => {
	    if (event && event.__cmAutofill) return;
	    captureField(event && event.target);
	  }, true);

	  document.addEventListener("blur", (event) => {
	    if (event && event.__cmAutofill) return;
	    captureField(event && event.target);
	  }, true);

  document.addEventListener("submit", (event) => {
    const form = event && event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const fields = form.querySelectorAll("input, textarea");
    for (const field of fields) {
      captureField(field);
    }
  }, true);
})();
"""
        await self._context.add_init_script(script=script)

    def _persist_login_form_memory(self) -> None:
        if not self._captured_login_fields:
            return
        existing = self._load_login_form_memory_snapshot()
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in existing:
            key = (
                str(item.get("origin", "")),
                str(item.get("path", "/")),
                str(item.get("fingerprint", "")),
            )
            if not key[0] or not key[2]:
                continue
            merged[key] = item

        for key, item in self._captured_login_fields.items():
            prior = merged.get(key)
            if prior is None or int(prior.get("updated_at", 0)) <= int(item.get("updated_at", 0)):
                merged[key] = item

        records = sorted(
            merged.values(),
            key=lambda record: int(record.get("updated_at", 0)),
            reverse=True,
        )
        # Keep recent records only, to bound file growth.
        records = records[:300]
        self._save_login_form_memory_snapshot(records)

    def _save_session_storage_snapshot(self, snapshot: dict[str, dict[str, str]]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_storage_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    async def _install_session_storage_init_script(self) -> None:
        if self._context is None:
            return
        snapshot = self._load_session_storage_snapshot()
        if not snapshot:
            return
        serialized = json.dumps(snapshot)
        script = (
            "(() => {"
            f"const persisted = {serialized};"
            "try {"
            "  const origin = window.location.origin;"
            "  const entries = persisted[origin];"
            "  if (!entries || typeof entries !== 'object') return;"
            "  for (const [k, v] of Object.entries(entries)) {"
            "    try { sessionStorage.setItem(k, String(v)); } catch (_) {}"
            "  }"
            "} catch (_) {}"
            "})();"
        )
        await self._context.add_init_script(script=script)

    async def _persist_session_storage_from_context(self) -> None:
        if self._context is None:
            return

        existing = self._load_session_storage_snapshot()
        updated: dict[str, dict[str, str]] = dict(existing)

        for page in self._context.pages:
            raw_url = (page.url or "").strip()
            parsed = urlparse(raw_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            origin = f"{parsed.scheme}://{parsed.netloc}"
            try:
                values = await page.evaluate(
                    """() => {
                        const out = {};
                        for (let i = 0; i < sessionStorage.length; i += 1) {
                            const key = sessionStorage.key(i);
                            if (key === null) continue;
                            const value = sessionStorage.getItem(key);
                            out[key] = value === null ? "" : value;
                        }
                        return out;
                    }"""
                )
            except Exception:
                continue

            if not isinstance(values, dict):
                continue
            cleaned: dict[str, str] = {}
            for key, value in values.items():
                if not isinstance(key, str):
                    continue
                if value is None:
                    cleaned[key] = ""
                elif isinstance(value, str):
                    cleaned[key] = value
                else:
                    cleaned[key] = str(value)
            if cleaned:
                updated[origin] = cleaned

        if updated != existing:
            self._save_session_storage_snapshot(updated)

    async def _persist_cookies(self) -> None:
        """Save all cookies (including session cookies) to disk."""
        if self._context is None:
            return
        try:
            cookies = await self._context.cookies()
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.cookies_path.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
        except Exception:
            pass

    async def _restore_cookies(self) -> None:
        """Restore previously saved cookies into the current context."""
        if self._context is None:
            return
        if not self.cookies_path.exists():
            return
        try:
            cookies = json.loads(self.cookies_path.read_text(encoding="utf-8"))
            if isinstance(cookies, list) and cookies:
                await self._context.add_cookies(cookies)
        except Exception:
            pass

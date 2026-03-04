from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import async_playwright

from browse_story import run_browse_story_mode
from execution_common import (
    COOKIES_FILE,
    SESSION_STORAGE_FILE,
    _STEALTH,
    _android_emulation_context_kwargs,
    _launch_context_kwargs,
)
from execution import (
    GLOBAL_LOGIN_INFO_FILE,
    LockRegistry,
    QueryOutcome,
    run_query_execution,
    write_results_file,
)
from planning import create_plan_for_query
from websites import (
    CHROMIUM_BROWSER,
    DEFAULT_BROWSER,
    WebsiteQuery,
    parse_websites_md,
    write_websites_md,
)


MODE_SCRAPE = "scrape"
MODE_SCRAPE_HEADLESS = "scrape-headless"
MODE_MANUAL = "manual"
MODE_BROWSE = "browse"
MODE_EDIT_LOGINS = "edit-logins"
MODE_EDIT_WEBSITES = "edit-websites"
SUPPORTED_MODES = [
    MODE_SCRAPE,
    MODE_SCRAPE_HEADLESS,
    MODE_MANUAL,
    MODE_BROWSE,
    MODE_EDIT_LOGINS,
    MODE_EDIT_WEBSITES,
]
LAST_MODE_FILE = Path.home() / ".computer_meister_last_mode"
DEFAULT_WEBSITES_PATH = Path("websites.md")


def _load_last_mode() -> str | None:
    if not LAST_MODE_FILE.exists():
        return None
    try:
        mode = LAST_MODE_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if mode in SUPPORTED_MODES:
        return mode
    return None


def _save_last_mode(mode: str) -> None:
    LAST_MODE_FILE.write_text(f"{mode}\n", encoding="utf-8")


def _resolve_mode(cli_mode: str | None) -> str:
    if cli_mode is not None:
        return cli_mode
    saved_mode = _load_last_mode()
    if saved_mode is not None:
        return saved_mode
    return MODE_SCRAPE


def _print_mode_banner(active_mode: str) -> None:
    others = [mode for mode in SUPPORTED_MODES if mode != active_mode]
    print(f"Active mode: {active_mode}", flush=True)
    print(f"Other available modes: {', '.join(others)}", flush=True)


def _run_single_query(
    query: WebsiteQuery,
    session_id: str,
    locks: LockRegistry,
    *,
    allow_manual_login: bool,
) -> QueryOutcome:
    base_dir = Path("site_data")
    query_dir = base_dir / query.dir_name
    query_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = query_dir / f"{session_id}_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results_path = query_dir / f"{session_id}_results.md"

    try:
        _plan_path, commands, _history = create_plan_for_query(query, session_id, query_dir)

        session_lock = locks.session_lock_for(query.session_dir_name)
        with session_lock:
            outcome = run_query_execution(
                query=query,
                commands=commands,
                artifact_dir=artifacts_dir,
                results_path=results_path,
                session_dir=base_dir / query.session_dir_name,
                login_prompt_lock=locks.login_prompt_lock,
                allow_manual_login=allow_manual_login,
            )
        return outcome
    except Exception as exc:  # noqa: BLE001
        body = f"FAIL\n{exc}"
        write_results_file(
            path=results_path,
            elapsed_ms=0,
            headless=True,
            help_used=False,
            body=body,
        )
        return QueryOutcome(
            query=query,
            success=False,
            skipped=False,
            results_path=results_path,
            pertinent_text=body,
            elapsed_ms=0,
            headless=True,
            help_used=False,
        )


def _print_completion(outcome: QueryOutcome) -> None:
    section = outcome.query.section_id
    if outcome.skipped:
        print(f"[{section}] SKIP ({outcome.results_path})", flush=True)
        return
    if not outcome.success:
        print(f"[{section}] FAIL ({outcome.results_path})", flush=True)
        return
    print(f"[{section}]", flush=True)
    print(outcome.pertinent_text.strip(), flush=True)


def _enabled_queries(queries: list[WebsiteQuery]) -> list[WebsiteQuery]:
    return [query for query in queries if not query.disabled]


def _run_scrape_mode(*, websites_path: Path, allow_manual_login: bool) -> None:
    session_id = str(int(time.time()))

    queries = parse_websites_md(websites_path)
    targets = _enabled_queries(queries)

    if not targets:
        print("No enabled queries found in websites.md", flush=True)
        return

    locks = LockRegistry()

    if len(targets) == 1:
        outcome = _run_single_query(targets[0], session_id, locks, allow_manual_login=allow_manual_login)
        _print_completion(outcome)
        return

    max_workers = min(8, len(targets))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_query,
                query,
                session_id,
                locks,
                allow_manual_login=allow_manual_login,
            ): query
            for query in targets
        }
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            _print_completion(outcome)


def _load_session_storage_snapshot(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
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


def _save_session_storage_snapshot(path: Path, snapshot: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


async def _install_session_storage_init_script(context: Any, session_storage_path: Path) -> None:
    snapshot = _load_session_storage_snapshot(session_storage_path)
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
    await context.add_init_script(script=script)


async def _install_human_login_stealth(context: Any) -> None:
    script = """
(() => {
    try {
        delete Object.getPrototypeOf(navigator).webdriver;
        Object.defineProperty(navigator, "webdriver", {
            get: () => undefined,
            configurable: true
        });
    } catch (_) {}

    try {
        if (!window.chrome || !window.chrome.runtime) {
            window.chrome = {
                runtime: { id: undefined },
                loadTimes: () => ({}),
                csi: () => ({}),
                app: { isInstalled: false }
            };
        }
    } catch (_) {}

    try {
        if (navigator.permissions && navigator.permissions.query) {
            const origQuery = navigator.permissions.query.bind(navigator.permissions);
            navigator.permissions.query = (params) =>
                (params && params.name === "notifications")
                    ? Promise.resolve({ state: "prompt", onchange: null })
                    : origQuery(params);
        }
    } catch (_) {}

    try { delete window.__playwright; } catch (_) {}
    try { delete window.__pw_manual; } catch (_) {}
    try { delete window.__playwright__binding__; } catch (_) {}
    try { delete window.__pwInitScripts; } catch (_) {}
})();
"""
    await context.add_init_script(script=script)


async def _persist_session_storage_from_context(context: Any, session_storage_path: Path) -> None:
    existing = _load_session_storage_snapshot(session_storage_path)
    updated: dict[str, dict[str, str]] = dict(existing)

    for page in context.pages:
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
        _save_session_storage_snapshot(session_storage_path, updated)


async def _restore_cookies(context: Any, cookies_path: Path) -> None:
    if not cookies_path.exists():
        return
    try:
        cookies = json.loads(cookies_path.read_text(encoding="utf-8"))
        if isinstance(cookies, list) and cookies:
            await context.add_cookies(cookies)
    except Exception:
        return


async def _persist_cookies(context: Any, cookies_path: Path) -> None:
    try:
        cookies = await context.cookies()
        cookies_path.parent.mkdir(parents=True, exist_ok=True)
        cookies_path.write_text(json.dumps(cookies, indent=2), encoding="utf-8")
    except Exception:
        return


async def _run_manual_session(*, query: WebsiteQuery, session_dir: Path) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    browser = query.effective_browser
    target_url = query.site if query.site.startswith(("http://", "https://")) else f"https://{query.site}"
    session_storage_path = session_dir / SESSION_STORAGE_FILE
    cookies_path = session_dir / COOKIES_FILE
    android_device = ""

    playwright = await async_playwright().start()
    context = None
    try:
        if query.android:
            android_device, _android_kwargs = _android_emulation_context_kwargs(playwright.devices)

        launch_kwargs = _launch_context_kwargs(
            browser=query.browser,
            headless=False,
            android=query.android,
            devices=playwright.devices,
        )

        if browser == CHROMIUM_BROWSER:
            context = await playwright.chromium.launch_persistent_context(
                str(session_dir),
                **launch_kwargs,
            )
        else:
            context = await playwright.firefox.launch_persistent_context(
                str(session_dir),
                **launch_kwargs,
            )

        if _STEALTH is not None:
            try:
                await _STEALTH.apply_stealth_async(context)
            except Exception:
                pass
        await _install_human_login_stealth(context)
        await _install_session_storage_init_script(context, session_storage_path)
        await _restore_cookies(context, cookies_path)
        page = context.pages[0] if context.pages else await context.new_page()
        try:
            await page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
        except Exception as exc:  # noqa: BLE001
            print(f"Navigation warning for {target_url}: {exc}", flush=True)

        print(
            f"Manual session for {query.section_id} using {browser}"
            + (f" ({android_device} emulation)." if query.android else "."),
            flush=True,
        )
        print("Interact in the browser window. Press ENTER here to close the session.", flush=True)
        await asyncio.to_thread(input)
    finally:
        if context is not None:
            try:
                await _persist_session_storage_from_context(context, session_storage_path)
            except Exception:
                pass
            try:
                await _persist_cookies(context, cookies_path)
            except Exception:
                pass
            try:
                await context.close()
            except Exception:
                pass
        await playwright.stop()


def _run_manual_mode(*, websites_path: Path) -> None:
    try:
        queries = parse_websites_md(websites_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not parse {websites_path}: {exc}", flush=True)
        return

    targets = _enabled_queries(queries)
    if not targets:
        print("No enabled queries found in websites.md", flush=True)
        return

    query = targets[0]
    base_dir = Path("site_data")
    session_dir = base_dir / query.session_dir_name
    asyncio.run(_run_manual_session(query=query, session_dir=session_dir))


def _normalize_site(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or text).strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _normalize_authenticated_url(value: Any) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return text


def _host_from_origin(origin: str) -> str:
    try:
        parsed = urlparse(origin)
    except Exception:
        return ""
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _mask_secret(secret: str) -> str:
    if not secret:
        return "<empty>"
    if len(secret) <= 2:
        return "*" * len(secret)
    return secret[:1] + ("*" * (len(secret) - 2)) + secret[-1:]


def _prompt_yes_no(prompt: str, *, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.", flush=True)


def _pick_from_list(options: list[str], *, prompt: str) -> str | None:
    if not options:
        return None
    for idx, value in enumerate(options, start=1):
        print(f"  {idx}) {value}", flush=True)
    while True:
        raw = input(prompt).strip()
        if raw.isdigit():
            picked = int(raw)
            if 1 <= picked <= len(options):
                return options[picked - 1]
        print("Invalid selection.", flush=True)


def _load_login_info(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                payload = dict(parsed)
        except Exception:
            payload = {}

    raw_records = payload.get("records")
    records: list[dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    if isinstance(raw_records, list):
        for item in raw_records:
            if not isinstance(item, dict):
                continue
            origin = item.get("origin")
            fingerprint = item.get("fingerprint")
            if not isinstance(origin, str) or not origin:
                continue
            if not isinstance(fingerprint, str) or not fingerprint:
                continue
            path_value = item.get("path")
            if not isinstance(path_value, str):
                path_value = "/"
            value = item.get("value")
            if value is None:
                value = ""
            if not isinstance(value, str):
                value = str(value)
            updated_at = item.get("updated_at")
            if not isinstance(updated_at, int):
                updated_at = now_ms
            records.append(
                {
                    "origin": origin,
                    "path": path_value,
                    "fingerprint": fingerprint,
                    "value": value,
                    "is_password": bool(item.get("is_password", False)),
                    "input_type": str(item.get("input_type", "")),
                    "updated_at": updated_at,
                }
            )

    credentials: dict[str, dict[str, Any]] = {}
    raw_credentials = payload.get("credentials")
    if isinstance(raw_credentials, dict):
        for site_key, item in raw_credentials.items():
            site = _normalize_site(str(site_key))
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
            entry: dict[str, Any] = {
                "username": username,
                "password": password,
                "updated_at": updated_at,
            }
            authenticated_url = _normalize_authenticated_url(item.get("authenticated_url"))
            if authenticated_url:
                entry["authenticated_url"] = authenticated_url
            credentials[site] = entry

    return payload, records, credentials


def _save_login_info(
    path: Path,
    payload: dict[str, Any],
    records: list[dict[str, Any]],
    credentials: dict[str, dict[str, Any]],
) -> None:
    out: dict[str, Any] = dict(payload)
    version = out.get("version", 1)
    try:
        out["version"] = int(version)
    except Exception:
        out["version"] = 1
    out["saved_at"] = datetime.now(timezone.utc).isoformat()
    out["records"] = records
    out["credentials"] = credentials
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def _record_likely_username(record: dict[str, Any]) -> bool:
    if bool(record.get("is_password", False)):
        return False
    fp = str(record.get("fingerprint", "")).lower()
    if any(token in fp for token in ("search", "coupon", "promo", "captcha", "otp", "2fa", "verification")):
        return False
    if any(token in fp for token in ("user", "email", "login", "account", "identifier", "phone")):
        return True
    input_type = str(record.get("input_type", "")).lower()
    return input_type in {"", "text", "email", "tel"}


def _apply_credentials_to_records(records: list[dict[str, Any]], site: str, username: str, password: str) -> None:
    now_ms = int(time.time() * 1000)
    for record in records:
        if _host_from_origin(str(record.get("origin", ""))) != site:
            continue
        if bool(record.get("is_password", False)):
            record["value"] = password
            record["updated_at"] = now_ms
            continue
        if _record_likely_username(record):
            record["value"] = username
            record["updated_at"] = now_ms


def _infer_credentials_from_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: int(item.get("updated_at", 0))):
        site = _host_from_origin(str(record.get("origin", "")))
        if not site:
            continue
        value = str(record.get("value", ""))
        if not value:
            continue
        entry = out.setdefault(site, {"username": "", "password": "", "updated_at": int(record.get("updated_at", 0))})
        entry["updated_at"] = max(int(entry.get("updated_at", 0)), int(record.get("updated_at", 0)))
        if bool(record.get("is_password", False)):
            entry["password"] = value
        elif _record_likely_username(record):
            entry["username"] = value
    return out


def _sites_from_websites_md(websites_path: Path) -> list[str]:
    try:
        queries = parse_websites_md(websites_path)
    except Exception:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        site = _normalize_site(query.site)
        if not site or site in seen:
            continue
        seen.add(site)
        out.append(site)
    return sorted(out)


def run_edit_logins_mode(websites_path: Path) -> None:
    payload, records, credentials = _load_login_info(GLOBAL_LOGIN_INFO_FILE)
    dirty = False

    while True:
        inferred = _infer_credentials_from_records(records)
        combined: dict[str, dict[str, Any]] = {**inferred, **credentials}

        print("\nLogin entries:", flush=True)
        if not combined:
            print("  (none)", flush=True)
        else:
            for site in sorted(combined):
                entry = combined[site]
                source = "explicit" if site in credentials else "inferred"
                username = str(entry.get("username", ""))
                password = str(entry.get("password", ""))
                authenticated_url = _normalize_authenticated_url(entry.get("authenticated_url"))
                authenticated_url_suffix = f" authenticated_url={authenticated_url}" if authenticated_url else ""
                print(
                    (
                        f"  - {site} [{source}] username={_mask_secret(username)} "
                        f"password={_mask_secret(password)}{authenticated_url_suffix}"
                    ),
                    flush=True,
                )

        print(
            "\nCommands: [v]iew, [a]dd, [e]dit, [s]ave+exit, [q]uit",
            flush=True,
        )
        cmd = input("Choice: ").strip().lower()

        if cmd in {"", "v", "view"}:
            continue

        if cmd in {"a", "add"}:
            website_sites = _sites_from_websites_md(websites_path)
            existing_with_credentials = {
                site
                for site, entry in combined.items()
                if str(entry.get("username", "")) and str(entry.get("password", ""))
            }
            available = [site for site in website_sites if site not in existing_with_credentials]
            if not available:
                print("No websites.md sites are missing credentials.", flush=True)
                continue
            print("\nPick a site to add credentials:", flush=True)
            picked_site = _pick_from_list(available, prompt="Select site number: ")
            if not picked_site:
                print("No site selected.", flush=True)
                continue

            username = input("Username: ").strip()
            password = input("Password: ").strip()
            if not username and not password:
                print("Username and password cannot both be empty.", flush=True)
                continue

            now_ms = int(time.time() * 1000)
            existing_entry = credentials.get(picked_site, {})
            authenticated_url = _normalize_authenticated_url(
                existing_entry.get("authenticated_url") if isinstance(existing_entry, dict) else ""
            )
            entry: dict[str, Any] = {
                "username": username,
                "password": password,
                "updated_at": now_ms,
            }
            if authenticated_url:
                entry["authenticated_url"] = authenticated_url
            credentials[picked_site] = entry
            _apply_credentials_to_records(records, picked_site, username, password)
            dirty = True
            print(f"Added credentials for {picked_site}.", flush=True)
            continue

        if cmd in {"e", "edit"}:
            editable_sites = sorted(combined)
            if not editable_sites:
                print("No credentials to edit.", flush=True)
                continue
            print("\nPick a site to edit:", flush=True)
            picked_site = _pick_from_list(editable_sites, prompt="Select site number: ")
            if not picked_site:
                print("No site selected.", flush=True)
                continue

            current = combined[picked_site]
            current_username = str(current.get("username", ""))
            current_password = str(current.get("password", ""))

            username_input = input(f"Username [{current_username}]: ").strip()
            password_input = input(f"Password [{_mask_secret(current_password)}]: ").strip()
            username = username_input if username_input else current_username
            password = password_input if password_input else current_password

            now_ms = int(time.time() * 1000)
            authenticated_url = _normalize_authenticated_url(current.get("authenticated_url"))
            entry: dict[str, Any] = {
                "username": username,
                "password": password,
                "updated_at": now_ms,
            }
            if authenticated_url:
                entry["authenticated_url"] = authenticated_url
            credentials[picked_site] = entry
            _apply_credentials_to_records(records, picked_site, username, password)
            dirty = True
            print(f"Updated credentials for {picked_site}.", flush=True)
            continue

        if cmd in {"s", "save"}:
            _save_login_info(GLOBAL_LOGIN_INFO_FILE, payload, records, credentials)
            print(f"Saved {GLOBAL_LOGIN_INFO_FILE}.", flush=True)
            return

        if cmd in {"q", "quit"}:
            if dirty and not _prompt_yes_no("Discard unsaved login changes?", default=False):
                continue
            print("Exited without saving.", flush=True)
            return

        print("Unknown command.", flush=True)


def _section_key_in_use(queries: list[WebsiteQuery], *, site: str, number: int, ignore_index: int | None = None) -> bool:
    for idx, query in enumerate(queries):
        if ignore_index is not None and idx == ignore_index:
            continue
        if query.site == site and query.number == number:
            return True
    return False


def _next_number_for_site(queries: list[WebsiteQuery], site: str) -> int:
    used = {query.number for query in queries if query.site == site}
    candidate = 0
    while candidate in used:
        candidate += 1
    return candidate


def _format_query_flags(query: WebsiteQuery) -> str:
    flags: list[str] = []
    if query.disabled:
        flags.append("disabled")
    if query.android:
        flags.append("android")
    elif query.browser == CHROMIUM_BROWSER:
        flags.append("chromium")
    if query.nofill:
        flags.append("nofill")
    if query.nocapture:
        flags.append("nocapture")
    return ", ".join(flags) if flags else "none"


def _print_websites(queries: list[WebsiteQuery]) -> None:
    print("\nwebsites.md sections:", flush=True)
    if not queries:
        print("  (none)", flush=True)
        return
    for idx, query in enumerate(queries, start=1):
        print(
            f"  {idx}) {query.section_id} | flags: {_format_query_flags(query)} | query: {query.query}",
            flush=True,
        )


def _pick_query_index(queries: list[WebsiteQuery]) -> int | None:
    if not queries:
        return None
    raw = input("Select section number: ").strip()
    if not raw.isdigit():
        print("Invalid number.", flush=True)
        return None
    value = int(raw)
    if not (1 <= value <= len(queries)):
        print("Out of range.", flush=True)
        return None
    return value - 1


def _edit_section(queries: list[WebsiteQuery], index: int) -> bool:
    changed = False
    query = queries[index]

    while True:
        print(
            f"\nEditing {query.section_id} | flags: {_format_query_flags(query)} | query: {query.query}",
            flush=True,
        )
        print(
            "Fields: [1]site [2]number [3]query [4]toggle-disabled "
            "[5]browser [6]toggle-android [7]toggle-nofill [8]toggle-nocapture [9]done",
            flush=True,
        )
        cmd = input("Field: ").strip()

        if cmd == "1":
            raw_site = input(f"Site [{query.site}]: ").strip()
            if not raw_site:
                continue
            new_site = _normalize_site(raw_site)
            if not new_site:
                print("Site cannot be empty.", flush=True)
                continue
            if _section_key_in_use(queries, site=new_site, number=query.number, ignore_index=index):
                print(f"Duplicate section header: {new_site} {query.number}", flush=True)
                continue
            query = replace(query, site=new_site)
            changed = True
            continue

        if cmd == "2":
            raw_number = input(f"Number [{query.number}]: ").strip()
            if not raw_number:
                continue
            if not raw_number.isdigit():
                print("Number must be an integer >= 0.", flush=True)
                continue
            new_number = int(raw_number)
            if _section_key_in_use(queries, site=query.site, number=new_number, ignore_index=index):
                print(f"Duplicate section header: {query.site} {new_number}", flush=True)
                continue
            query = replace(query, number=new_number)
            changed = True
            continue

        if cmd == "3":
            new_query = input(f"Query [{query.query}]: ").strip()
            if not new_query:
                print("Query cannot be empty.", flush=True)
                continue
            query = replace(query, query=new_query)
            changed = True
            continue

        if cmd == "4":
            query = replace(query, disabled=not query.disabled)
            changed = True
            continue

        if cmd == "5":
            if query.browser == DEFAULT_BROWSER:
                query = replace(query, browser=CHROMIUM_BROWSER, android=False)
            else:
                query = replace(query, browser=DEFAULT_BROWSER)
            changed = True
            continue

        if cmd == "6":
            if query.android:
                query = replace(query, android=False)
            else:
                query = replace(query, android=True, browser=DEFAULT_BROWSER)
            changed = True
            continue

        if cmd == "7":
            query = replace(query, nofill=not query.nofill)
            changed = True
            continue

        if cmd == "8":
            query = replace(query, nocapture=not query.nocapture)
            changed = True
            continue

        if cmd == "9":
            queries[index] = query
            return changed

        print("Unknown selection.", flush=True)


def run_edit_websites_mode(websites_path: Path) -> None:
    if websites_path.exists():
        try:
            queries = parse_websites_md(websites_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not parse {websites_path}: {exc}", flush=True)
            return
    else:
        queries = []

    dirty = False

    while True:
        _print_websites(queries)
        print(
            "\nCommands: [a]dd, [e]dit, [r]emove, [s]ave+exit, [q]uit",
            flush=True,
        )
        cmd = input("Choice: ").strip().lower()

        if cmd in {"", "v", "view"}:
            continue

        if cmd in {"a", "add"}:
            site = _normalize_site(input("Site (example.com): ").strip())
            if not site:
                print("Site is required.", flush=True)
                continue

            default_number = _next_number_for_site(queries, site)
            raw_number = input(f"Number [{default_number}]: ").strip()
            if raw_number:
                if not raw_number.isdigit():
                    print("Number must be an integer >= 0.", flush=True)
                    continue
                number = int(raw_number)
            else:
                number = default_number

            if _section_key_in_use(queries, site=site, number=number):
                print(f"Duplicate section header: {site} {number}", flush=True)
                continue

            query_text = input("Query: ").strip()
            if not query_text:
                print("Query is required.", flush=True)
                continue

            android = _prompt_yes_no("Use android emulation? (implies chromium everywhere)", default=False)
            use_chromium = False
            if not android:
                use_chromium = _prompt_yes_no("Use chromium browser?", default=False)
            disabled = _prompt_yes_no("Mark disabled?", default=False)
            nofill = _prompt_yes_no("Set nofill?", default=False)
            nocapture = _prompt_yes_no("Set nocapture?", default=False)

            queries.append(
                WebsiteQuery(
                    site=site,
                    number=number,
                    query=query_text,
                    disabled=disabled,
                    browser=CHROMIUM_BROWSER if use_chromium else DEFAULT_BROWSER,
                    android=android,
                    nofill=nofill,
                    nocapture=nocapture,
                )
            )
            dirty = True
            continue

        if cmd in {"e", "edit"}:
            index = _pick_query_index(queries)
            if index is None:
                continue
            dirty = _edit_section(queries, index) or dirty
            continue

        if cmd in {"r", "remove"}:
            index = _pick_query_index(queries)
            if index is None:
                continue
            query = queries[index]
            if not _prompt_yes_no(f"Remove section '{query.section_id}'?", default=False):
                continue
            del queries[index]
            dirty = True
            continue

        if cmd in {"s", "save"}:
            write_websites_md(websites_path, queries)
            print(f"Saved {websites_path}.", flush=True)
            return

        if cmd in {"q", "quit"}:
            if dirty and not _prompt_yes_no("Discard unsaved websites changes?", default=False):
                continue
            print("Exited without saving.", flush=True)
            return

        print("Unknown command.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Computer Meister")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=SUPPORTED_MODES,
        help=(
            "Mode: scrape, scrape-headless, manual, browse, edit-logins, edit-websites. "
            "If omitted, uses the last selected mode (default: scrape)."
        ),
    )
    args = parser.parse_args()

    mode = _resolve_mode(args.mode)
    if args.mode is not None:
        _save_last_mode(mode)
    _print_mode_banner(mode)

    websites_path = DEFAULT_WEBSITES_PATH

    if mode == MODE_BROWSE:
        run_browse_story_mode(websites_path=websites_path)
        return

    if mode == MODE_EDIT_LOGINS:
        run_edit_logins_mode(websites_path=websites_path)
        return

    if mode == MODE_EDIT_WEBSITES:
        run_edit_websites_mode(websites_path=websites_path)
        return

    if mode == MODE_SCRAPE_HEADLESS:
        _run_scrape_mode(websites_path=websites_path, allow_manual_login=False)
        return

    if mode == MODE_MANUAL:
        _run_manual_mode(websites_path=websites_path)
        return

    _run_scrape_mode(websites_path=websites_path, allow_manual_login=True)


if __name__ == "__main__":
    main()

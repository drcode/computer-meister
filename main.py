from __future__ import annotations

import argparse
import concurrent.futures
import time
from pathlib import Path

from browse_story import run_browse_story_mode
from execution import LockRegistry, QueryOutcome, run_query_execution, write_results_file
from planning import create_plan_for_query
from websites import WebsiteQuery, parse_websites_md


def _run_single_query(query: WebsiteQuery, session_id: str, locks: LockRegistry) -> QueryOutcome:
    base_dir = Path("site_data")
    query_dir = base_dir / query.dir_name
    query_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = query_dir / f"{session_id}_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results_path = query_dir / f"{session_id}_results.md"

    try:
        _plan_path, commands, _history = create_plan_for_query(query, session_id, query_dir)

        site_lock = locks.site_lock_for(query.site)
        with site_lock:
            outcome = run_query_execution(
                query=query,
                commands=commands,
                artifact_dir=artifacts_dir,
                results_path=results_path,
                session_dir=base_dir / query.session_dir_name,
                login_prompt_lock=locks.login_prompt_lock,
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
            results_path=results_path,
            pertinent_text=body,
            elapsed_ms=0,
            headless=True,
            help_used=False,
        )


def _print_completion(outcome: QueryOutcome) -> None:
    section = outcome.query.section_id
    if not outcome.success:
        print(f"[{section}] FAIL ({outcome.results_path})", flush=True)
        return
    print(f"[{section}]", flush=True)
    print(outcome.pertinent_text.strip(), flush=True)


def _enabled_queries(queries: list[WebsiteQuery]) -> list[WebsiteQuery]:
    return [query for query in queries if not query.disabled]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch web retrieval tool")
    parser.add_argument(
        "--browse",
        action="store_true",
        help="Browse an existing query artifact session and render an HTML story",
    )
    parser.add_argument("--serial", action="store_true", help="Process all queries serially")
    parser.add_argument(
        "--websites",
        default="websites.md",
        help="Path to websites.md",
    )
    args = parser.parse_args()

    websites_path = Path(args.websites)

    if args.browse:
        run_browse_story_mode(websites_path=websites_path)
        return

    session_id = str(int(time.time()))

    queries = parse_websites_md(websites_path)
    targets = _enabled_queries(queries)

    if not targets:
        print("No enabled queries found in websites.md", flush=True)
        return

    locks = LockRegistry()

    if args.serial or len(targets) == 1:
        for query in targets:
            outcome = _run_single_query(query, session_id, locks)
            _print_completion(outcome)
        return

    max_workers = min(8, len(targets))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_query, query, session_id, locks): query
            for query in targets
        }
        for future in concurrent.futures.as_completed(futures):
            outcome = future.result()
            _print_completion(outcome)


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from typing import Any

from playwright.async_api import Page

from execution_common import (
    COMPUTER_USE_MODEL,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    _dump_json,
    _normalize_key,
    _plan_command_line,
    _plan_command_payload,
)
from planning import PlanCommand


def _computer_action_to_plan_command(action: dict[str, Any]) -> PlanCommand:
    action_type = str(action.get("type", "")).strip()

    if action_type == "click":
        button = str(action.get("button", "left")).lower()
        if button != "left":
            raise RuntimeError(
                "unsupported click button for plan mapping "
                f"(expected left): {_dump_json(action)}"
            )
        return PlanCommand("click", (int(action["x"]), int(action["y"])))

    if action_type == "type":
        text = action.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"type action missing string text: {_dump_json(action)}")
        return PlanCommand("type", (text,))

    if action_type == "keypress":
        keys = action.get("keys")
        if not isinstance(keys, list) or not keys:
            raise RuntimeError(f"keypress action missing keys list: {_dump_json(action)}")
        return PlanCommand("keypress", tuple(str(key) for key in keys))

    if action_type == "wait":
        return PlanCommand("wait", (int(action.get("ms", 500)),))

    if action_type == "move":
        return PlanCommand("wait", (500,))

    if action_type == "scroll":
        scroll_x = int(action.get("scroll_x", 0))
        if scroll_x != 0:
            raise RuntimeError(
                "unsupported horizontal scroll for plan mapping: "
                f"{_dump_json(action)}"
            )
        return PlanCommand("vscroll", (int(action.get("scroll_y", 0)),))

    raise RuntimeError(
        "unsupported computer action for plan mapping: "
        f"{_dump_json(action)}"
    )


async def _execute_plan_interaction(page: Page, command: PlanCommand, *, allow_text_entry: bool) -> None:
    if command.name == "click":
        await page.mouse.click(int(command.args[0]), int(command.args[1]), button="left")
        return

    if command.name == "type":
        if not allow_text_entry:
            raise RuntimeError("type command encountered before enable_text_entry")
        await page.keyboard.type(str(command.args[0]))
        return

    if command.name == "keypress":
        for key in command.args:
            await page.keyboard.press(_normalize_key(str(key)))
        return

    if command.name == "wait":
        await asyncio.sleep(max(0, int(command.args[0])) / 1000.0)
        return

    if command.name == "vscroll":
        await page.mouse.wheel(0, int(command.args[0]))
        return

    raise RuntimeError(f"unsupported plan interaction command: {command.name}")


class ExplorationMixin:
    async def _run_exploration(self, instruction: str, max_steps: int) -> None:
        max_steps = max(1, min(max_steps, 80))
        prompt = (
            "Explore this website to gather information for later answering.\n"
            f"Query target: {self.query.query}\n"
            f"Exploration goal: {instruction}\n"
            "Stop once you have gathered enough context and then return a concise summary.\n"
        )
        if not self.allow_text_entry:
            prompt += "Do not enter text into inputs or search fields.\n"

        initial = await self.recorder.capture(self.page, source="explore_start", metadata={"instruction": instruction})
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

        exploration_loop_index = self._next_llm_index("exploration_loop")
        initial_request = {
            "model": COMPUTER_USE_MODEL,
            "tools": tools,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{initial_b64}"},
                    ],
                }
            ],
            "truncation": "auto",
        }
        try:
            response = self.openai_client.responses.create(**initial_request)
            self._write_llm_call_log("exploration_loop", initial_request, response, index=exploration_loop_index)
        except Exception as exc:  # noqa: BLE001
            self._write_llm_call_log(
                "exploration_loop",
                initial_request,
                None,
                index=exploration_loop_index,
                error=exc,
            )
            raise

        steps_out: list[dict[str, Any]] = []
        final_text = ""

        for step_idx in range(max_steps):
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
                # Screenshot is a protocol handshake action: respond with the latest frame.
                self._print_exploration_step(
                    step_idx=step_idx,
                    max_steps=max_steps,
                    message="screenshot handshake",
                )
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
                self._print_exploration_step(
                    step_idx=step_idx,
                    max_steps=max_steps,
                    message=_plan_command_line(plan_command),
                )
                action_error = None
                try:
                    await _execute_plan_interaction(self.page, plan_command, allow_text_entry=self.allow_text_entry)
                except Exception as exc:  # noqa: BLE001
                    action_error = str(exc)

                captured = await self.recorder.capture(
                    self.page,
                    source="explore_action",
                    metadata={
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
                    self._print_exploration_step(
                        step_idx=step_idx,
                        max_steps=max_steps,
                        message=f"error: {action_error}",
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
                    "exploration_loop",
                    followup_request,
                    response,
                    index=self._next_llm_index("exploration_loop"),
                )
            except Exception as exc:  # noqa: BLE001
                self._write_llm_call_log(
                    "exploration_loop",
                    followup_request,
                    None,
                    index=self._next_llm_index("exploration_loop"),
                    error=exc,
                )
                raise

        run_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instruction": instruction,
            "max_command_steps": max_steps,
            "allow_text_entry": self.allow_text_entry,
            "final_text": final_text,
            "steps": steps_out,
        }
        self.exploration_runs.append(run_record)
        exploration_path = self.artifact_dir / "exploration_steps.json"
        exploration_path.write_text(
            json.dumps({"runs": self.exploration_runs}, indent=2),
            encoding="utf-8",
        )

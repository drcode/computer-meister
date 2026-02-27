from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from ai import openai
from execution_common import ANSWER_MODEL, _artifact_index, _responses_text


class ArtifactRecorder:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.counter = 0
        self.events: list[dict[str, Any]] = []

    async def capture(self, page: Page, *, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        idx = self.counter
        self.counter += 1

        screenshot_name = f"screenshot_{idx}.png"
        html_name = f"page_{idx}.html"
        screenshot_path = self.artifact_dir / screenshot_name
        html_path = self.artifact_dir / html_name

        screenshot_bytes = await page.screenshot(path=str(screenshot_path))
        main_html = await page.content()
        main_html = _filter_artifact_html(main_html)

        html_path.write_text(main_html, encoding="utf-8")

        event = {
            "index": idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "screenshot": screenshot_name,
            "page_html": html_name,
        }
        if metadata:
            event["metadata"] = metadata

        self.events.append(event)
        return {
            "event": event,
            "screenshot_bytes": screenshot_bytes,
            "screenshot_path": screenshot_path,
            "html_path": html_path,
        }

    def latest_screenshots(self, count: int = 4) -> list[Path]:
        pairs = sorted(self.artifact_dir.glob("screenshot_*.png"), key=_artifact_index)
        return pairs[-count:]

    def latest_pages(self, count: int = 4) -> list[Path]:
        pairs = sorted(self.artifact_dir.glob("page_*.html"), key=_artifact_index)
        return pairs[-count:]


def _filter_artifact_html(raw_html: str) -> str:
    try:
        import lxml.html
        from lxml import etree

        doc = lxml.html.fromstring(raw_html)
        selectors = [
            "script",
            "style",
            "noscript",
            "svg",
            "link",
            "object",
            "embed",
            "applet",
            "img",
        ]
        for selector in selectors:
            for el in doc.cssselect(selector):
                el.getparent().remove(el)

        for comment in doc.iter(etree.Comment):
            comment.getparent().remove(comment)

        return lxml.html.tostring(doc, encoding="unicode", pretty_print=False)
    except Exception:
        return raw_html


def preprocess_html(raw_html: str) -> str:
    """
    Strip non-content fat from HTML while preserving DOM structure
    enough for XPath selectors to work on the cleaned result.
    """
    import lxml.html
    from lxml import etree

    doc = lxml.html.fromstring(raw_html)

    # 1. Remove elements that are never informational.
    tags_to_remove = [
        "script",
        "style",
        "noscript",
        "svg",
        "path",
        "link",
        "meta",
        "object",
        "embed",
        "applet",
        "picture > source",  # Keep <img> but drop responsive source hints.
    ]
    for tag in tags_to_remove:
        for el in doc.cssselect(tag):
            el.getparent().remove(el)

    # 2. Remove HTML comments.
    for comment in doc.iter(etree.Comment):
        comment.getparent().remove(comment)

    # 3. Strip non-structural attributes (biggest token saver).
    keep_attrs = {
        "src",
        "alt",
        "title",
        "id",
        "name",
        "type",
        "value",
        "placeholder",
        "aria-label",
        "role",
        "datetime",
    }
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        removable = [a for a in el.attrib if a not in keep_attrs]
        for attr in removable:
            del el.attrib[attr]

    # 4. Remove hidden elements (inline style display:none / visibility:hidden).
    for el in doc.iter():
        style = el.get("style", "")
        compact = style.replace(" ", "")
        if "display:none" in compact or "visibility:hidden" in compact:
            el.getparent().remove(el)

    # 5. Remove empty non-void elements to reduce wrapper noise.
    void_tags = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
    for el in reversed(list(doc.iter())):
        if not isinstance(el.tag, str):
            continue
        if el.tag.lower() in void_tags:
            continue
        if el.getparent() is None:
            continue
        if len(el) == 0 and not el.text_content().strip():
            el.getparent().remove(el)

    # 6. Remove <head> entirely (metadata, not content).
    title_text: str | None = None
    head = doc.find(".//head")
    if head is not None:
        title_el = head.find("title")
        title_text = title_el.text_content() if title_el is not None else None
        head.getparent().remove(head)

    # 7. Collapse whitespace in text nodes.
    for el in doc.iter():
        if el.text:
            el.text = re.sub(r"\s+", " ", el.text).strip()
        if el.tail:
            el.tail = re.sub(r"\s+", " ", el.tail).strip()

    result = lxml.html.tostring(doc, encoding="unicode", pretty_print=False)

    if title_text:
        result = f"<!-- Page title: {title_text} -->\n{result}"

    return result


class AnsweringMixin:
    async def _answer_query_text(self, instruction: str) -> str:
        pages = self.recorder.latest_pages(4)
        if not pages:
            capture = await self.recorder.capture(self.page, source="answer_query_text_autocapture")
            pages = [capture["html_path"]]

        latest_page = max(pages, key=_artifact_index)
        try:
            import html2text

            converter = html2text.HTML2Text()
            converter.body_width = 0
            converter.ignore_links = True
        except Exception as exc:
            raise RuntimeError("html2text is required for answer_query_text but could not be imported") from exc

        page_chunks: list[str] = []
        for page_path in pages:
            raw = page_path.read_text(encoding="utf-8", errors="replace")
            if page_path == latest_page:
                try:
                    processed = preprocess_html(raw)
                except Exception:
                    processed = raw
                page_chunks.append(f"FILE: {page_path.name}\nCURRENT_HTML:\n{processed}")

            try:
                page_text = converter.handle(raw)
            except Exception as exc:
                raise RuntimeError(
                    f"html2text conversion failed for {page_path.name} in answer_query_text"
                ) from exc
            page_chunks.append(f"FILE: {page_path.name}\nPAGE_TEXT_HTML2TEXT:\n{page_text}")

        prompt = (
            "Answer the query strictly from the provided webpage snapshots.\n"
            "Each file includes PAGE_TEXT_HTML2TEXT for all pages, and CURRENT_HTML for one of the pages. You can use any or all pages to answer the query.\n"
            f"Original query: {self.query.query}\n"
            f"Answer instruction: {instruction}\n\n"
            "If the evidence is insufficient or contradictory, start with FAIL and explain what is missing.\n\n"
            + "\n\n".join(page_chunks)
        )
        request = {
            "model": ANSWER_MODEL,
            "system": "Provide a concise factual answer grounded in the supplied snapshots.",
            "max_output_tokens": 1200,
            "temperature": 0,
            "prompt": prompt,
        }

        try:
            response = openai(
                prompt,
                system="Provide a concise factual answer grounded in the supplied snapshots.",
                max_output_tokens=1200,
                temperature=0,
            )
            self._write_llm_call_log("answer_query_text", request, response)
        except Exception as exc:  # noqa: BLE001
            self._write_llm_call_log("answer_query_text", request, None, error=exc)
            raise
        return str(response).strip()

    async def _answer_query_images(self, instruction: str) -> str:
        screenshots = self.recorder.latest_screenshots(4)
        if not screenshots:
            capture = await self.recorder.capture(self.page, source="answer_query_images_autocapture")
            screenshots = [capture["screenshot_path"]]

        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Answer the query from the provided webpage screenshots.\n"
                    f"Original query: {self.query.query}\n"
                    f"Answer instruction: {instruction}\n"
                    "If evidence is insufficient, start with FAIL and explain what is missing."
                ),
            }
        ]

        for image_path in screenshots:
            b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                }
            )

        request = {
            "model": ANSWER_MODEL,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 1200,
        }
        try:
            response = self.openai_client.responses.create(**request)
            self._write_llm_call_log("answer_query_images", request, response)
        except Exception as exc:  # noqa: BLE001
            self._write_llm_call_log("answer_query_images", request, None, error=exc)
            raise
        text = _responses_text(response).strip()
        if text:
            return text
        return "FAIL\nNo text response was returned from image analysis."

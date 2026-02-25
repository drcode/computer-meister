import lxml.html
from lxml.html.clean import Cleaner
import re

def preprocess_html(raw_html: str) -> str:
    """
    Strip non-content fat from HTML while preserving DOM structure
    enough for XPath selectors to work on the cleaned result.
    """
    doc = lxml.html.fromstring(raw_html)

    # 1. Remove elements that are never informational
    tags_to_remove = [
        "script", "style", "noscript", "svg", "path", "link", "meta",
        "object", "embed", "applet",
        "picture > source",  # keep <img> but drop responsive source hints
    ]
    for tag in tags_to_remove:
        for el in doc.cssselect(tag):
            el.getparent().remove(el)

    # 2. Remove HTML comments
    from lxml import etree
    for comment in doc.iter(etree.Comment):
        comment.getparent().remove(comment)

    # 3. Strip non-structural attributes (biggest token saver)
    KEEP_ATTRS = {"src", "alt", "title", "id", "name", "type",
                  "value", "placeholder", "aria-label", "role", "datetime"}
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        removable = [a for a in el.attrib if a not in KEEP_ATTRS]
        for attr in removable:
            del el.attrib[attr]

    # 4. Remove hidden elements (inline style display:none / visibility:hidden)
    for el in doc.iter():
        style = el.get("style", "")
        if "display:none" in style.replace(" ", "") or \
           "visibility:hidden" in style.replace(" ", ""):
            el.getparent().remove(el)

    # 5. Remove empty non-void elements to reduce wrapper noise.
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
    for el in reversed(list(doc.iter())):
        if not isinstance(el.tag, str):
            continue
        if el.tag.lower() in VOID_TAGS:
            continue
        if el.getparent() is None:
            continue
        if len(el) == 0 and not el.text_content().strip():
            el.getparent().remove(el)

    # 6. Remove <head> entirely (metadata, not content) — optional
    head = doc.find(".//head")
    if head is not None:
        # Optionally preserve <title> text before removing
        title_el = head.find("title")
        title_text = title_el.text_content() if title_el is not None else None
        head.getparent().remove(head)

    # 7. Collapse whitespace in text nodes
    for el in doc.iter():
        if el.text:
            el.text = re.sub(r'\s+', ' ', el.text).strip()
        if el.tail:
            el.tail = re.sub(r'\s+', ' ', el.tail).strip()

    result = lxml.html.tostring(doc, encoding="unicode", pretty_print=False)

    # Optionally prepend title
    if title_text:
        result = f"<!-- Page title: {title_text} -->\n{result}"

    return result

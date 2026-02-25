from pathlib import Path
import signal
import sys
import webbrowser

from playwright.sync_api import sync_playwright


def capture_homepage_screenshot(url: str, output_path: Path) -> None:
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True, timeout=8_000)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.set_default_timeout(10_000)
        page.set_default_navigation_timeout(10_000)
        page.goto(url, wait_until="domcontentloaded", timeout=10_000)
        # Give the page a brief moment to finish first paint.
        page.wait_for_timeout(1500)
        page.screenshot(path=str(output_path), full_page=True, timeout=8_000)
        browser.close()


def display_image(image_path: Path) -> None:
    try:
        from PIL import Image

        Image.open(image_path).show()
    except Exception:
        webbrowser.open(image_path.resolve().as_uri())


if __name__ == "__main__":
    def _alarm_handler(signum, frame):
        raise TimeoutError("Exceeded 30 second total runtime limit")

    # Hard cap: process is terminated if it exceeds 30 seconds on Unix-like systems.
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(30)

    screenshot_path = Path("fidelity_homepage.png")
    try:
        capture_homepage_screenshot("https://www.fidelity.com", screenshot_path)
    except Exception as exc:
        print(f"ERROR: failed to capture fidelity.com screenshot: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    display_image(screenshot_path)

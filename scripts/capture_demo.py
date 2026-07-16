"""Capture assets/demo.png from the running app.

Streamlit renders over a WebSocket after first paint, so a naive headless screenshot
catches the skeleton loader instead of the app. This waits for content that only exists
once the pipeline has actually run.

Start the app first:
    BIOENHANCE_RETRIEVER=tfidf streamlit run app.py --server.port 8501 --server.headless true

Then (needs `pip install playwright && playwright install chromium`):
    python scripts/capture_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

URL = "http://localhost:8501/?name=itraconazole&run=1&backend=rulebased&top_k=6"
OUT = Path(__file__).resolve().parents[1] / "assets" / "demo.png"


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("pip install playwright && playwright install chromium")

    OUT.parent.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        page.goto(URL, wait_until="networkidle", timeout=60_000)

        # Only present once the assessment has rendered — not on the skeleton loader.
        page.wait_for_selector("text=Ranked strategies", timeout=90_000)
        page.wait_for_timeout(2500)  # let Plotly finish drawing

        # Collapse the sidebar so the image is about the science, not the input form.
        try:
            page.click("[data-testid='stSidebarCollapseButton']", timeout=4000)
            page.wait_for_timeout(1200)
        except Exception:  # noqa: BLE001 - sidebar controls vary across Streamlit versions
            pass

        page.screenshot(path=str(OUT), full_page=True)
        browser.close()

    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

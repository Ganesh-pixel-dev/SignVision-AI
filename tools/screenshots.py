"""Take screenshots of the running call app with Edge and fake camera flags.

Usage: python tools/screenshots.py http://127.0.0.1:5001 prefix [join]
Writes docs/screenshots/<prefix>-desktop.png and <prefix>-phone.png.
With "join" it clicks through joining a room first, using a fake camera.
"""

import os
import sys

from playwright.sync_api import sync_playwright

EDGE = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots")


def main(url, prefix, join=False):
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=EDGE, args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        for name, size in (("desktop", (1440, 900)), ("phone", (390, 844))):
            ctx = browser.new_context(viewport={"width": size[0], "height": size[1]},
                                      permissions=["camera", "microphone"])
            page = ctx.new_page()
            page.goto(url)
            page.wait_for_timeout(800)
            if join:
                page.click("#joinBtn")
                page.wait_for_timeout(2500)
            page.screenshot(path=os.path.join(OUT, f"{prefix}-{name}.png"), full_page=True)
            width = page.evaluate("document.documentElement.scrollWidth")
            print(name, "scrollWidth", width, "viewport", size[0])
            ctx.close()
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "join")

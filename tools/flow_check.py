"""Drive two browser pages through a call against a running server and screenshot it.

Usage: python tools/flow_check.py http://127.0.0.1:5001
Uses Edge with a fake camera, so no hand is ever detected: this checks the
UI states and signalling, not recognition quality.
"""

import os
import sys

from playwright.sync_api import sync_playwright

EDGE = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots")


def main(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=EDGE, args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        pages = []
        for _ in range(2):
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["camera", "microphone"])
            pages.append(ctx.new_page())
        a, b = pages
        for page in pages:
            page.goto(url)
            page.click("#joinBtn")
            page.wait_for_selector("#call:not([hidden])")
        a.wait_for_timeout(3000)
        print("room status:", a.inner_text("#roomStatus"))
        a.click("#signBtn")
        a.wait_for_timeout(1000)
        print("mode a:", a.inner_text("#modeLine"))
        print("mode b:", b.inner_text("#modeLine"))
        b.evaluate("socket.emit('speech_text', {text: 'hello and thank you'})")
        a.wait_for_timeout(2500)
        print("heard:", a.inner_text("#heard"), "| label:", a.inner_text("#skeletonLabel"))
        a.screenshot(path=os.path.join(OUT, "after-call-signer-desktop.png"), full_page=True)
        b.screenshot(path=os.path.join(OUT, "after-call-listener-desktop.png"), full_page=True)
        third = browser.new_context(permissions=["camera", "microphone"]).new_page()
        third.goto(url)
        third.click("#joinBtn")
        third.wait_for_selector("#alerts .alert")
        print("third person sees:", third.inner_text("#alerts"))
        b.set_viewport_size({"width": 390, "height": 844})
        b.wait_for_timeout(500)
        b.screenshot(path=os.path.join(OUT, "after-call-listener-phone.png"), full_page=True)
        print("phone scrollWidth", b.evaluate("document.documentElement.scrollWidth"))
        a.click("#leaveBtn")
        b.wait_for_timeout(500)
        print("b after a left:", b.inner_text("#roomStatus"))
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1])

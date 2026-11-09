"""Open the hosted dashboard in a headless browser and wake it if Streamlit Cloud put it to sleep.

Used by .github/workflows/keep-awake.yml; reads the address from the APP_URL environment variable.
"""

import os

from playwright.sync_api import sync_playwright

url = os.environ["APP_URL"]
with sync_playwright() as p:
    page = p.chromium.launch().new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(10_000)
    wake = page.get_by_role("button", name="get this app back up")
    if wake.count():
        print("App was asleep; waking it up.")
        wake.first.click()
        page.wait_for_timeout(90_000)  # give it time to boot
    else:
        print("App is awake.")

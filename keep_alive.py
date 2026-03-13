from playwright.sync_api import sync_playwright
import time

APPS = [
    "https://custom-comparison-app.streamlit.app/",
    "https://customizable-team-savant-page-creator.streamlit.app/",
    "https://customizable-player-savant-page.streamlit.app/",
    "https://hitterleaderboardapp.streamlit.app/",
    "https://hitter-year-over-year.streamlit.app/",
    "https://custompitchercomparisonapp.streamlit.app/",
    "https://pitcher-savant-page.streamlit.app/",
    "https://team-pitching-savant.streamlit.app/",
    "https://pitcher-leaderboard.streamlit.app/",
    "https://pitcher-year-over-year.streamlit.app/",
]

def wake_app(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Waking {url}...")
        page.goto(url, wait_until="networkidle")
        time.sleep(15)  # let WebSocket fully establish
        print(f"Done: {url}")
        browser.close()

for app in APPS:
    wake_app(app)
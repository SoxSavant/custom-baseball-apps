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
    "https://hitter-stat-filter.streamlit.app/",
    "https://pitcher-stat-filter.streamlit.app/",
    "https://hitter-season.streamlit.app/",
    "https://pitcher-season.streamlit.app/",
    "https://hitter-league-leaders.streamlit.app/",
    "https://pitcher-league-leaders.streamlit.app/",
    "https://hitters-per-team.streamlit.app/",
    "https://pitchers-per-team.streamlit.app/"
]

def wake_app(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Waking {url}...")
        try:
            page.goto(url, wait_until="networkidle", timeout = 60000)
            time.sleep(15)  
            print(f"Done: {url}")
        except Exception as e:
            print(f"Failed to wake {url}: {e}")
        finally:
            browser.close()

for app in APPS:
    wake_app(app)
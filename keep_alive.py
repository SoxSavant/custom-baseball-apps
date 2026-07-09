from playwright.sync_api import sync_playwright

APPS = [
    "https://custom-comparison-app.streamlit.app/",
    "https://customizable-player-savant-page.streamlit.app/",
    "https://hitterleaderboardapp.streamlit.app/",
    "https://hitter-year-over-year.streamlit.app/",
    "https://custompitchercomparisonapp.streamlit.app/",
    "https://pitcher-savant-page.streamlit.app/",
    "https://pitcher-leaderboard.streamlit.app/",
    "https://pitcher-year-over-year.streamlit.app/",
    "https://hitter-stat-filter.streamlit.app/",
    "https://pitcher-stat-filter.streamlit.app/",
    "https://hitter-season.streamlit.app/",
    "https://pitcher-season.streamlit.app/",
    "https://hitter-league-leaders.streamlit.app/",
    "https://pitcher-league-leaders.streamlit.app/",
    "https://hitters-per-team.streamlit.app/",
    "https://pitchers-per-team.streamlit.app/",
    "https://hitter-indiv-yoy.streamlit.app/",
    "https://pitcher-indiv-yoy.streamlit.app/",
    "https://hitting-database.streamlit.app/",
    "https://pitching-database.streamlit.app/",
    "https://compositedatabase.streamlit.app/",
    "https://stat-correlation.streamlit.app/",
    "https://compositedatabase2.streamlit.app/",
    "https://stat-trajectory.streamlit.app/",
]

WAKE_BUTTON_SELECTOR = "button:has-text('get this app back up')"

def wake_app(page, url):
    print(f"Visiting {url}...")
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # Check if the app is asleep and needs the wake button clicked
        wake_button = page.locator(WAKE_BUTTON_SELECTOR)
        try:
            wake_button.wait_for(state="visible", timeout=5000)
            print(f"  App was asleep, clicking wake button...")
            wake_button.click()
            # Give the app time to actually spin back up
            page.wait_for_timeout(30000)
        except Exception:
            # No wake button found — app was already awake
            page.wait_for_load_state("networkidle", timeout=15000)

        print(f"Done: {url}")
    except Exception as e:
        print(f"Failed to wake {url}: {e}")

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for app in APPS:
            wake_app(page, app)
        browser.close()

if __name__ == "__main__":
    main()
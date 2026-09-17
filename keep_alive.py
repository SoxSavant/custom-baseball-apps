from playwright.sync_api import sync_playwright
import re

APPS = [
    "https://hitter-league-leaders.streamlit.app/",
    "https://pitcher-league-leaders.streamlit.app/",
    "https://compositedatabase.streamlit.app/",
    "https://compositedatabase2.streamlit.app/",
    "https://stat-correlation.streamlit.app/",
    "https://stat-trajectory.streamlit.app/",
    "https://warleaders.streamlit.app/",
    "https://custom-comparison.streamlit.app/",
    "https://custom-leaderboard.streamlit.app/",
    "https://soxsavant-database.streamlit.app/",
    "https://players-per-team.streamlit.app/",
    "https://custom-savant-page.streamlit.app/",
    "https://season-counter.streamlit.app/",
    "https://stat-filter.streamlit.app/",
    "https://league-yoy.streamlit.app/",
    "https://player-yoy.streamlit.app/",
    "https://streak-finder.streamlit.app/",
    "https://player-ranks.streamlit.app/",
]

# Loose, case-insensitive match — Streamlit's copy has shifted before
# ("Yes, get this app back up!" / "Get this app back up!" etc.)
WAKE_BUTTON_SELECTOR = "button:has-text('back up')"
APP_CONTAINER_SELECTOR = "[data-testid='stAppViewContainer'], [data-testid='stApp']"

# A generic headless UA is one of the easiest bot-detection tells.
# Give the context a normal desktop Chrome UA to reduce the odds of a
# challenge page instead of the real app.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def app_is_live(page) -> bool:
    """Real check, not an inference: did the actual Streamlit app render?"""
    try:
        page.wait_for_selector(APP_CONTAINER_SELECTOR, timeout=10000, state="attached")
        return True
    except Exception:
        return False


def wake_app(page, url):
    print(f"Visiting {url}...")
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        wake_button = page.locator(WAKE_BUTTON_SELECTOR)
        button_visible = False
        try:
            wake_button.wait_for(state="visible", timeout=10000)
            button_visible = True
        except Exception:
            pass

        if button_visible:
            print("  Asleep — clicking wake button...")
            wake_button.click()
            # Poll for the real app instead of a blind fixed sleep.
            for attempt in range(6):  # up to ~90s total
                page.wait_for_timeout(15000)
                if app_is_live(page):
                    print(f"  Woke up after ~{(attempt + 1) * 15}s")
                    return
            print(f"  WARNING: clicked wake but app never rendered — {url}")
            _dump_diagnostics(page, url)
            return

        # No wake button seen. Verify this actually means "awake",
        # rather than assuming it.
        if app_is_live(page):
            print("  Already awake.")
        else:
            print(f"  WARNING: no wake button AND no app content — likely "
                  f"blocked/challenge page or changed markup — {url}")
            _dump_diagnostics(page, url)

    except Exception as e:
        print(f"  FAILED: {url}: {e}")


def _dump_diagnostics(page, url):
    """Save a screenshot + title so failures are debuggable from CI artifacts
    instead of just a 'Done' log line that hides the real problem."""
    safe_name = re.sub(r"\W+", "_", url)
    try:
        page.screenshot(path=f"debug_{safe_name}.png", full_page=True)
        print(f"    page title: {page.title()!r}")
        print(f"    saved screenshot debug_{safe_name}.png")
    except Exception as e:
        print(f"    (could not capture diagnostics: {e})")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        for app in APPS:
            wake_app(page, app)
        browser.close()


if __name__ == "__main__":
    main()

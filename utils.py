from datetime import date

def get_dynamic_min_pa(year):
    # Only apply to current season
    if year == 2020:
        return 186
    if year != 2026:
        return 502

    opening_day = date(2026, 3, 26)
    today = date.today()

    days_since = (today - opening_day).days
    min_pa = max(10, int(days_since * 2.5))

    return min(min_pa, 502)

def get_dynamic_min_ip(year):
    if year != 2026:
        return 162

    opening_day = date(2026, 3, 26)
    today = date.today()

    days_since = (today - opening_day).days

    # Approx: 1 game per day → 1 IP per game
    min_ip = max(5, days_since * 1.0)

    return min(min_ip, 162)

def get_percentile_min_pa(year):
    if year == 2020:
        return 126
    if year != 2026:
        return 340  # your full-season cutoff

    opening_day = date(2026, 3, 26)
    today = date.today()
    days_since = (today - opening_day).days

    # 2.1 PA per team game pace → ~340 over full season
    min_pa = max(20, days_since * 2.1)

    return min(min_pa, 340)

def get_percentile_min_ip(year):
    if year == 2020:
        return 15
    if year != 2026:
        return 40  # your full-season cutoff

    opening_day = date(2026, 3, 26)
    today = date.today()
    days_since = (today - opening_day).days

    # ~0.25 IP per team game → ~40 over season
    min_ip = max(3, days_since * 0.25)

    return min(min_ip, 40)
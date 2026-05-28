from datetime import date
from p_utils import normalize_team

current_year = date.today().year

def get_dynamic_min_pa(year):
    # Only apply to current season
    if year == 2020:
        return 186
    if year < current_year:
        return 502

    opening_day = date(current_year, 3, 26) # rough estimate
    today = date.today()

    days_since = (today - opening_day).days
    min_pa = max(10, int(days_since * 2.5))

    return min(min_pa, 502)

def get_dynamic_min_ip(year):
    if year < current_year:
        return 162

    opening_day = date(current_year, 3, 26) # rough estimate
    today = date.today()

    days_since = (today - opening_day).days

    # Approx: 0.8 games per day → 1 IP per game
    min_ip = max(5, days_since * 0.8)

    return min(min_ip, 162)

def get_percentile_min_pa(year):
    if year == 2020:
        return 126
    if year < current_year:
        return 340  

    opening_day = date(current_year, 3, 26) # rough estimate
    today = date.today()
    days_since = (today - opening_day).days

    # 2.1 PA per team game pace → ~340 over full season
    min_pa = max(20, days_since * 2.1)

    return min(min_pa, 340)

def get_percentile_min_ip(year):
    if year == 2020:
        return 15
    if year < current_year:
        return 40  

    opening_day = date(current_year, 3, 26)
    today = date.today()
    days_since = (today - opening_day).days

    # ~0.25 IP per team game → ~40 over season
    min_ip = max(3, days_since * 0.25)

    return min(min_ip, 40)

TEAM_MLB_IDS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111,
    "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115,
    "CHW": 145, "DET": 116, "HOU": 117, "KCR": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158,
    "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SDP": 135, "SEA": 136,
    "SFG": 137, "STL": 138, "TBR": 139, "TEX": 140,
    "TOR": 141, "WSN": 120,
}

DIVISION_TEAMS = {
    "AL East":    {"BAL", "BOS", "NYY", "TBR", "TOR"},
    "AL Central": {"CHW", "CLE", "DET", "KCR", "MIN"},
    "AL West":    {"HOU", "LAA", "OAK", "SEA", "TEX"},
    "NL East":    {"ATL", "MIA", "NYM", "PHI", "WSN"},
    "NL Central": {"CHC", "CIN", "MIL", "PIT", "STL"},
    "NL West":    {"ARI", "COL", "LAD", "SDP", "SFG"},
}

ALL_DIVISIONS = list(DIVISION_TEAMS.keys())

def get_team_division(abbrev: str) -> str:
    a = normalize_team(abbrev)
    for div, teams in DIVISION_TEAMS.items():
        if a in teams:
            return div
    return ""

def get_team_logo_url(abbrev: str) -> str:
    mlb_id = TEAM_MLB_IDS.get(normalize_team(abbrev))
    if mlb_id:
        return f"https://www.mlbstatic.com/team-logos/{mlb_id}.svg"
    return ""




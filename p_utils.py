from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import requests

TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}

start_year = 1901

LOCAL_BWAR_FILE = Path(__file__).with_name("warpitchers.txt")

HEADSHOT_BASE_SILO = (
    "https://img.mlbstatic.com/mlb-photos/image/upload"
    "/d_people:generic:headshot:67:current.png"
    "/w_240,q_auto:best,f_auto/v1/people/{mlbam}/headshot/silo/current"
)
HEADSHOT_BASE_67 = (
    "https://img.mlbstatic.com/mlb-photos/image/upload"
    "/d_people:generic:headshot:67:current.png"
    "/w_240,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current"
)
HEADSHOT_PLACEHOLDER = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0nMjQwJyBoZWlnaHQ9JzI0MCcgdmlld0JveD0nMCAwIDI0MCAyNDAnIHhtbG5zPSdodHRwOi8v"
    "d3d3LnczLm9yZy8yMDAwL3N2Zyc+CjxyZWN0IHdpZHRoPScyNDAnIGhlaWdodD0nMjQwJyBmaWxsPScjZWVmJy8+"
    "CjxjaXJjbGUgY3g9JzEyMCcgY3k9Jzk1JyByPSc1NScgZmlsbD0nI2RkZScvPgo8Y2lyY2xlIGN4PScxMjAnIGN5"
    "PSc4NScgcj0nNDInIGZpbGw9JyNmZmYnIHN0cm9rZT0nI2NjYycvPgo8cGF0aCBkPSdNMTIwIDE1MGMtMzAgMC01"
    "NSAyNS01NSA1NXMzNSAxNS41IDU1IDE1LjUgNTUtMTUuNSA1NS0xNS41LTM1LTU1LTU1LTU1eicgZmlsbD0nI2Nj"
    "YycvPgo8L3N2Zz4="
)

TEAM_ALIASES = {"ATH": "OAK", "ATH/OAK": "OAK", "OAK/ATH": "OAK"}

STAT_DISPLAY_NAMES = {
    "HardHit%": "Hard Hit%",
}



STAT_ALLOWLIST = [
    "fWAR", "bWAR",
    "ERA", "xERA", "FIP", "xFIP", "vFA", "K%", "BB%", "K-BB%", "IP", "G", "GS",
    "Barrel%", "HardHit%", "EV", "GB%", "HR/9", "BABIP", "LOB%", "HR/FB",
    "SV", "AVG", "WHIP", "ERA-", "FIP-", "SIERA",
    "Chase%", "Whiff%", "WPA", "Clutch",
    "SO", "BB", "HBP", "HR", "QS", "CG", "ShO"
]

STAT_DEFAULTS = {
    "fWAR": 4.0, "bWAR": 4.0, "ERA": 3.00, "xERA": 3.00, "FIP": 3.00, "xFIP": 3.00,
    "WHIP": 1.10, "ERA-": 80.0, "FIP-": 80.0, "SIERA": 3.50,
    "IP": 162.0, "G": 30.0, "GS": 25.0, "W": 12.0, "L": 10.0,
    "SV": 20.0, "SO": 180.0, "BB": 50.0,
    "K/9": 10.0, "BB/9": 2.5, "HR/9": 1.0,
    "K%": 25.0, "BB%": 7.0, "K-BB%": 18.0,
    "Barrel%": 6.0, "HardHit%": 35.0, "EV": 88.0,
    "Chase%": 32.0, "Whiff%": 25.0,
    "GB%": 50.0, "HR/FB": 10.0,
    "BABIP": 0.280, "WPA": 2.0, "Clutch": 1.0,
    "CG": 1.0, "ShO": 1.0,
}

EVERY_STAT_PRESET = ["fWAR", "bWAR", "W-L", "vFA",
        "ERA", "xERA", "FIP", "xFIP", "IP", "G", "GS", "SO", "BB", "HBP", "HR", "K/9",
        "BB/9", "HR/9", "BABIP", "LOB%", "HR/FB", "QS", "CG", "ShO",
        "SV", "K%", "BB%", "K-BB%", "AVG", "WHIP", "ERA-", "FIP-",
        "Barrel%", "HardHit%", "EV", "GB/FB", "GB%", "FB%", "SIERA",
        "Chase%", "Whiff%", "WPA", "Clutch" ]

SUM_STATS = {
    "G", "GS", "HR", "BB", "SO", "HBP", "QS", "CG", "ShO", "SV", "WPA", "W", "L", "fWAR", "bWAR"
}
RATE_STATS = {
    "ERA", "xERA", "FIP", "xFIP", "K/9", "BB/9", "HR/9", "BABIP", "LOB%", "HR/FB",
    "K%", "BB%", "K-BB%", "AVG", "WHIP", "Barrel%", "HardHit%", "EV",
    "GB/FB", "GB%", "FB%", "SIERA", "Chase%", "Whiff%", "Clutch",
    "ERA-", "FIP-", "vFA",
}

PCT_STATS = {
    "K%", "BB%", "K-BB%", "Chase%", "Whiff%", "Barrel%", "HardHit%",
    "GB%", "FB%", "LOB%", "HR/FB",
}

label_map = {
    "EV": "Avg Exit Velo",
    "HardHit%": "Hard Hit%",
}

lower_better = {
    "ERA", "xERA", "FIP", "xFIP", "SIERA", "BB", "HBP", "HR",
    "BB/9", "HR/9", "BABIP", "HR/FB", "BB%", "AVG", "WHIP",
    "ERA-", "FIP-", "Barrel%", "HardHit%", "EV",
}


def normalize_team(team: str) -> str:
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)


def get_team_display(team_value: str) -> str:
    t = str(team_value).strip()
    if t == "- - -":
        return "2+ Teams"
    return normalize_team(t)


def _silo_exists(mlbam: int) -> bool:
    try:
        url = HEADSHOT_BASE_SILO.format(mlbam=mlbam)
        r = requests.head(url, timeout=3)
        return r.status_code == 200 and int(r.headers.get("content-length", 0)) > 10000
    except Exception:
        return False

def get_headshot(row: pd.Series) -> str:
    val = row.get("MLBAMID")
    if val is not None and pd.notna(val):
        mlbam = int(val)
        if _silo_exists(mlbam):
            return HEADSHOT_BASE_SILO.format(mlbam=mlbam)
        return HEADSHOT_BASE_67.format(mlbam=mlbam)
    return HEADSHOT_PLACEHOLDER
TEAM_OPTIONS = {
    "all": "All Teams",
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "CHW": "CHW", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "ATH": "ATH",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
}

TEAMS = {
    "ARI": "Arizona Diamondbacks",   "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",      "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",           "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",    "COL": "Colorado Rockies",
    "CHW": "Chicago White Sox",      "DET": "Detroit Tigers",
    "HOU": "Houston Astros",         "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",     "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",          "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",        "NYM": "New York Mets",
    "NYY": "New York Yankees",       "OAK": "Oakland Athletics",
    "ATH": "Athletics",
    "PHI": "Philadelphia Phillies",  "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",       "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants",   "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",         "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",      "WSN": "Washington Nationals",
}

# ─────────────────────────────────────────────
#  IP helpers
# ─────────────────────────────────────────────

def ip_to_outs(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    try:
        v = float(value)
    except Exception:
        return np.nan
    innings = int(np.floor(v))
    fractional = v - innings
    if abs(fractional - 0.1) < 0.05:
        outs_extra = 1
    elif abs(fractional - 0.2) < 0.05:
        outs_extra = 2
    else:
        outs_extra = min(max(int(round(fractional * 3)), 0), 2)
    return innings * 3 + outs_extra


def outs_to_ip(outs: float) -> float:
    if pd.isna(outs):
        return np.nan
    innings = int(float(outs) // 3)
    remainder = int(round(float(outs) % 3))
    return innings + remainder / 10

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FWAR", "BWAR"}:
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"EV", "vFA"}:
        return f"{float(val):.1f}"

    if upper_stat in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"

    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "SIERA", "K/9", "BB/9", "HR/9", "GB/FB", "HR/FB"}:
        return f"{float(val):.2f}"

    if upper_stat == "WHIP":
        return f"{float(val):.3f}"

    if upper_stat == "IP":
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"ERA-", "FIP-"}:
        return f"{int(round(float(val)))}"

    if upper_stat in {"BABIP", "AVG"}:
        return f"{float(val):.3f}".lstrip("0") or ".000"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat or "BB%" in stat
        or "Chase" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"

    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def format_stat_yoy(stat: str, val, show_sign: bool = False) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FWAR", "BWAR"}:
        v = float(val)
        formatted = f"{int(round(abs(v)))}.0" if abs(v - round(v)) < 1e-9 else f"{abs(v):.1f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"EV", "vFA"}:
        v = float(val)
        formatted = f"{abs(v):.1f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"WPA", "CLUTCH"}:
        v = float(val)
        return f"+{v:.2f}" if show_sign and v > 0 else f"{v:.2f}"

    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "SIERA", "K/9", "BB/9", "HR/9", "GB/FB"}:
        v = float(val)
        formatted = f"{abs(v):.2f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat == "WHIP":
        v = float(val)
        formatted = f"{abs(v):.3f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat == "IP":
        v = float(val)
        formatted = f"{int(round(abs(v)))}.0" if abs(v - round(v)) < 1e-9 else f"{abs(v):.1f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"ERA-", "FIP-"}:
        v = int(round(float(val)))
        return f"+{v}" if show_sign and v > 0 else f"{v}"

    if upper_stat in {"BABIP", "AVG"}:
        v = float(val)
        formatted = f"{abs(v):.3f}".lstrip("0") or ".000"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat or "BB%" in stat
        or "Chase" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        formatted = f"{abs(v):.1f}%"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{abs(v):.1f}%" if v < 0 else formatted

    v = float(val)
    formatted = f"{abs(v):.0f}" if abs(v - round(v)) < 1e-6 else f"{abs(v):.1f}"
    if show_sign and v > 0:
        return f"+{formatted}"
    return f"-{formatted}" if v < 0 else formatted
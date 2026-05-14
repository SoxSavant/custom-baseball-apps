from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import requests
import boto3
import os

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
    "https://img.mlbstatic.com/mlb-photos/image/upload"
    "/w_240,q_auto:best,f_auto/people/generic/headshot/67/current.png"
)

TEAM_ALIASES = {"ATH": "OAK", "ATH/OAK": "OAK", "OAK/ATH": "OAK"}

STAT_DISPLAY_NAMES = {
    "HardHit%": "Hard Hit%",
}



STAT_ALLOWLIST = [
    "fWAR", "bWAR",
    "ERA", "xERA", "FIP", "xFIP", "ERA-xERA","vFA", "K%", "BB%", "K-BB%", "IP", 
    "Chase%", "Whiff%", "G", "GS",
    "Barrel%", "HardHit%", "EV", "GB%", "K/9","BB/9","K/BB","HR/9", "BABIP", "LOB%", "HR/FB",
    "SV", "AVG", "WHIP", "ERA-", "FIP-", "SIERA",
     "WPA", "Clutch",
    "SO", "BB", "HBP", "HR", "QS", "CG", "ShO", "ER", "TBF", "fWAR/200", "bWAR/200"
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
    "fWAR/200": 5.0,
    "bWAR/200": 5.0,
    "ERA-xERA": 0.5,
}

EVERY_STAT_PRESET = ["fWAR", "bWAR", "W-L", "vFA",
        "ERA", "xERA", "FIP", "xFIP", "ERA-xERA","IP", "G", "GS", "SO", "BB", "HBP", "HR", "K/9",
        "BB/9", "HR/9", "BABIP", "LOB%", "HR/FB", "QS", "CG", "ShO",
        "SV", "K%", "BB%", "K-BB%", "BB/9","HR/9","K/BB","AVG", "WHIP", "ERA-", "FIP-",
        "Barrel%", "HardHit%", "EV", "GB/FB", "GB%", "FB%", "SIERA",
        "Chase%", "Whiff%", "WPA", "Clutch","fWAR/200", "bWAR/200"]



SUM_STATS = {
    "G", "GS", "HR", "BB", "SO", "HBP", "QS", "CG", "ShO", "SV", "WPA", "W", "L", "fWAR", "bWAR", "TBF",
}
RATE_STATS = {
    "ERA", "xERA", "FIP", "xFIP", "K/9", "BB/9", "HR/9", "BABIP", "LOB%", "HR/FB",
    "K%", "BB%", "K-BB%", "AVG", "WHIP", "Barrel%", "HardHit%", "EV",
    "GB/FB", "GB%", "FB%", "SIERA", "Chase%", "Whiff%", "Clutch",
    "ERA-", "FIP-", "vFA","BB/9","HR/9","K/BB","fWAR/200", "bWAR/200", "ERA-xERA",
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
    "ERA-", "FIP-", "Barrel%", "HardHit%", "EV", "HR/9","BB/9","ERA-xERA"
}


def normalize_team(team: str) -> str:
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)


def get_team_display(team_value: str) -> str:
    t = str(team_value).strip()
    if t == "- - -":
        return "2+ Teams"
    return normalize_team(t)


def _url_has_real_image(url: str) -> bool:
    try:
        r = requests.head(url, timeout=3)
        return r.status_code == 200 and int(r.headers.get("content-length", 0)) > 10000
    except Exception:
        return False

def get_headshot(row: pd.Series) -> str:
    val = row.get("MLBAMID")
    if val is not None and pd.notna(val):
        mlbam = int(val)
        for url_template in (HEADSHOT_BASE_SILO, HEADSHOT_BASE_67):
            url = url_template.format(mlbam=mlbam)
            if _url_has_real_image(url):
                return url
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


def get_player_id_by_name(name: str, year: int) -> int | None:
    df = load_final_year(year)
    if df is None or df.empty or "Name" not in df.columns:
        return None
    match = df[df["Name"].str.strip() == name.strip()]
    if match.empty:
        match = df[df["Name"].str.lower().str.strip() == name.lower().strip()]
    if match.empty:
        return None
    ids = match["PlayerId"].dropna()
    return int(ids.iloc[0]) if not ids.empty else None

def aggregate_player_group(grp: pd.DataFrame, start_year: int = 2015) -> dict:
    result: dict = {}

    result["Name"] = str(grp["Name"].iloc[0])
    result["PlayerId"] = grp["PlayerId"].iloc[0]
    result["MLBAMID"] = grp["MLBAMID"].iloc[0]

    teams = grp["Team"].astype(str).tolist()
    result["Team"] = get_team_display_multiseason(teams)

    outs_series = pd.to_numeric(grp["IP"], errors="coerce").apply(ip_to_outs)
    ip_outs_total = outs_series.sum(skipna=True)
    result["IP"] = outs_to_ip(ip_outs_total)

    weight = outs_series.fillna(0)
    weight_total = weight.sum()

    if "TBF" in grp.columns and not grp["TBF"].isna().all():
        tbf_total = pd.to_numeric(grp["TBF"], errors="coerce").sum(skipna=True)
    else:
        tbf_total = ip_outs_total + sum(
        pd.to_numeric(grp[c], errors="coerce").sum(skipna=True)
        for c in ("H", "BB", "HBP") if c in grp.columns
    )

    for col in grp.columns:
        if not pd.api.types.is_numeric_dtype(grp[col]) or col in {"PlayerId", "MLBAMID", "Season", "IP"}:
            continue
        series = pd.to_numeric(grp[col], errors="coerce")
        if series.isna().all():
            continue
        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in RATE_STATS and weight_total > 0:
            result[col] = (series * weight).sum(skipna=True) / weight_total
        else:
            result[col] = series.mean(skipna=True)

    ip_innings = ip_outs_total / 3.0
    bb, so, er, fwar, bwar = (pd.to_numeric(result.get(c), errors="coerce") for c in ("BB", "SO", "ER", "fWAR", "bWAR"))

    if pd.notna(er) and ip_innings > 0:
        result["ERA"] = (er / ip_innings) * 9
    if pd.notna(bb) and tbf_total > 0:
        result["BB%"] = (bb / tbf_total) * 100
    if pd.notna(so) and tbf_total > 0:
        result["K%"] = (so / tbf_total) * 100
    if pd.notna(bb) and ip_innings > 0:
        result["BB/9"] = (bb / ip_innings) * 9
    if pd.notna(so) and ip_innings > 0:
        result["K/9"] = (so / ip_innings) * 9
    if pd.notna(fwar) and ip_innings > 0:
        result["fWAR/200"] = fwar / ip_innings * 200
    if pd.notna(bwar) and ip_innings > 0:
        result["bWAR/200"] = bwar / ip_innings * 200

    return result

def get_team_display_multiseason(teams: list[str]) -> str:
    if any(get_team_display(t) == "2+ Teams" for t in teams):
        return "2+ Teams"
    normalized = {normalize_team(t) for t in teams if str(t).strip() and str(t).strip() != "- - -"}
    if len(normalized) > 1:
        return "2+ Teams"
    return normalized.pop() if normalized else "N/A"

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FWAR", "BWAR","FWAR/200", "BWAR/200"}:
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"EV", "vFA"}:
        return f"{float(val):.1f}"

    if upper_stat in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"

    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "SIERA", "K/9", "BB/9", "HR/9", "GB/FB", "HR/FB","ERA-XERA"}:
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

    if upper_stat in {"FWAR", "BWAR","FWAR/200", "BWAR/200"}:
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

    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "SIERA", "K/9", "BB/9", "HR/9", "GB/FB","ERA-XERA"}:
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

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

bucket = "sports-analytics-files"

@st.cache_data(show_spinner=False, ttl=900)
def load_final_year(year: int) -> pd.DataFrame:
    key = f"processed/pitching_final_{year}.csv"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj["Body"])
        df["Season"] = year
        return df
    except Exception:
        return pd.DataFrame()
    

# old loading function
    """@st.cache_data(show_spinner=False, ttl=900)
def load_final_year(year: int) -> pd.DataFrame:
    path = f"data/final/pitching_final_{year}.csv"
    try:
        df = pd.read_csv(path)
        df["Season"] = year
    
        return df
    except Exception:
        return pd.DataFrame()"""
    
STAT_PRESETS = {
    "Default": [
        "fWAR", "bWAR",  "GS","ERA", "xERA", "FIP", "IP",
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
    "Statcast": [
        "fWAR", "xERA", "vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
    "Standard": [
        "fWAR", "bWAR", "W-L", "ERA", "G", "GS", "IP", "AVG", "WHIP", "HR/9",
    ], 
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": ["fWAR"],
    "Player A leads": [],
    "Player B leads": [],
    "Player C leads": [],
    "Player D leads": [],
    "Player E leads": [],
}

STAT_PRESETS_SAVANT = {
    "Default": [
        "fWAR", "bWAR",  "ERA", "xERA", "FIP", "IP",
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
    "Statcast": [
        "fWAR", "xERA", "vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
    "Standard": [
        "fWAR", "bWAR", "ERA", "GS", "IP", "AVG", "WHIP", "HR/9", "K/BB",
    ],
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": ["fWAR"],
}

STAT_PRESETS_YOY = {
    "Statcast": [
         "xERA", "vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
    "Stat Mix": [
        "fWAR", "bWAR", "GS", "IP", "ERA",  "FIP", 
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
    "Standard": [
        "fWAR", "bWAR", "W-L", "ERA", "G", "GS", "IP", "AVG", "WHIP", "HR/9",
    ], 
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": ["fWAR"],
    "Only Improvements": [],
    "Only Regressions": [],
}
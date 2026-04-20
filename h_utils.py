from pathlib import Path
import streamlit as st
import pandas as pd
import requests
import boto3
import os

TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}

start_year = 1901

STAT_ALLOWLIST = [
    "fWAR", "bWAR", "Off", "Def", "BsR", "Barrel%", "HardHit%", "EV", "maxEV",
    "wRC+", "wOBA", "xwOBA", "wOBA-xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "TB", "H",
    "1B", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "BB/K","Chase%", "Whiff%", "WPA", "Clutch",
    "FRV", "OAA", "DRS", "FRM", "TZ","Swing%", "Z-Swing%",
    "O-Contact%", "Z-Contact%", "Zone%", "BatSpd", "Squared-Up%", "Inn",
]

SUM_STATS = {
    "G", "PA", "AB", "R", "H", "1B", "2B", "3B", "HR", "RBI", "SB",
    "BB", "IBB", "SO", "HBP", "SF", "SH", "XBH", "TB",
    "fWAR", "bWAR", "Off", "Def", "BsR",
    "DRS", "OAA", "FRV",
    "WPA", "FRM", "TZ",
}

RATE_STATS = {
    "AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "BABIP",
    "K%", "BB%", "K-BB%", "O-Swing%", "Whiff%",
    "Barrel%", "HardHit%",
    "EV","BB/K", "ISO", "BatSpd",
    "wRC+", "Clutch", "Chase%", "Swing%", "Z-Swing%",
    "O-Contact%", "Z-Contact%", "Zone%", "wOBA-xwOBA",
    "Squared-Up%"
}

MAX_STATS = {"maxEV"}

EVERY_STAT_PRESET = [
    "bWAR", "fWAR", "G", "AB", "PA",  "SB", "HR", "RBI", "XBH",
    "AVG", "OBP", "SLG", "OPS", "ISO", "BABIP",
    "wRC+", "Off", "BsR", "Def", "OAA", "FRV", "FRM", "wOBA",
    "xwOBA", "xBA", "xSLG", "EV", "maxEV", "Barrel%", "HardHit%",
    "Chase%", "Whiff%", "K%", "BB%", "BB/K","BB", "IBB", "SO",
    "H", "1B", "2B", "3B",  "TB", "R",
    "K-BB%", "DRS", "WPA", "Clutch", "Swing%", "Z-Swing%",
    "O-Contact%","Z-Contact%","Whiff%","Zone%","BatSpd", "wOBA-xwOBA", "TZ",
]

STAT_DEFAULTS = {
    "HR": 30, "SB": 30, "RBI": 100, "R": 100, "H": 150,
    "fWAR": 5.0, "bWAR": 5.0, "wRC+": 130, "wOBA": 0.370, "OPS": 0.900,
    "xwOBA": 0.370, "xBA": 0.280, "xSLG": 0.480,
    "AVG": 0.300, "OBP": 0.370, "SLG": 0.500, "ISO": 0.200,
    "K%": 20.0, "BB%": 10.0, "Barrel%": 12.0, "HardHit%": 45.0,
    "EV": 92.0, "BB": 60, "IBB": 10, "SO": 100, "PA": 502, "AB": 450, "BB/K": 1.0,
    "2B": 30, "1B": 100, "3B": 5, "XBH": 50, "TB": 300, "G": 140,
    "Clutch": 1.0, "FRV": 10, "OAA": 10, "DRS": 10,
    "Chase%": 25.0, "Whiff%": 20.0,
    "Off": 10.0, "Def": 5.0, "BsR": 3.0,
    "BABIP": 0.320, "WPA": 2.0, "wOBA-xwOBA": 0.020,
    "FRM": 5,
    "Swing%": 45.0, "Z-Swing%": 65.0, "O-Contact%": 65.0,
    "Z-Contact%": 85.0, "Zone%": 45.0,
    "maxEV": 112.0, "BatSpd": 73.0,
    "TZ": 10,
    "Squared-Up%": 25.0,
}

STAT_DISPLAY_NAMES = {
    "HardHit%": "Hard Hit%",
    "EV": "Avg EV",
    "BatSpd": "Bat Speed"
}

STATCAST_RATE_STATS = {"xwOBA", "xBA", "xSLG", "EV", "Barrel%", "HardHit%", "Squared-Up%"}

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


label_map = {
    "HardHit%": "Hard Hit%",
    "EV": "Avg EV",
    "BatSpd": "Bat Speed"
}
lower_better = {"K%", "Chase%", "Whiff%","SO"}


TEAM_ALIASES = {
    "ATH": "OAK",
    "ATH/OAK": "OAK",
    "OAK/ATH": "OAK",
}

POSITION_OPTIONS = {
    "all": "All Positions",
    "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
    "LF": "LF", "CF": "CF", "RF": "RF", "OF": "OF", "DH": "DH",
}

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

def normalize_team(team: str) -> str:
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)

def get_team_display(team_value: str) -> str:
    """
    Simple rule:
      - '- - -' means player was on 2+ teams → '2+ Teams'
      - Otherwise show the (normalized) team abbreviation
    """
    t = str(team_value).strip()
    if t == "- - -":
        return "2+ Teams"
    return normalize_team(t)

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FRV", "OAA", "DRS","TZ"}:
        return f"{int(round(float(val)))}"

    if upper_stat in {"WAR", "BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR", "MAXEV", "BATSPD","FRM"}:
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"WPA", "CLUTCH","BB/K"}:
        return f"{float(val):.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO", "WOBA-XWOBA"}:
        return f"{float(val):.3f}".lstrip("0") or ".000"

    if upper_stat in {"WRC+", "OPS+"}:
        return f"{int(round(float(val)))}"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
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

    if upper_stat in {"FRV", "OAA", "DRS","TZ"}:
        v = int(round(float(val)))
        return f"+{v}" if show_sign and v > 0 else f"{v}"

    if upper_stat in {"BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR", "MAXEV", "BATSPD", "FRM"}:
        v = float(val)
        formatted = f"{int(round(abs(v)))}.0" if abs(v - round(v)) < 1e-9 else f"{abs(v):.1f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"WPA", "CLUTCH","BB/K"}:
        v = float(val)
        return f"+{v:.2f}" if show_sign and v > 0 else f"{v:.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO", "WOBA-XWOBA"}:
        v = float(val)
        formatted = f"{abs(v):.3f}".lstrip("0") or ".000"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"WRC+"}:
        v = int(round(float(val)))
        return f"+{v}" if show_sign and v > 0 else f"{v}"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
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

def apply_dh_override(df):
    if "Pos" not in df.columns or "PA" not in df.columns or "Inn" not in df.columns or "Season" not in df.columns:
        print(df.columns)
        return df

    df = df.copy()


    # row-level DH eligibility (THIS is the key fix)
    eligible = df["Season"] >= 1973

    is_pitcher = df["Pos"].astype(str).str.upper().eq("P")
    eligible &= ~is_pitcher

    pa = pd.to_numeric(df["PA"], errors="coerce").fillna(0)
    inn = pd.to_numeric(df["Inn"], errors="coerce").fillna(0)

    estimated = (pa / 4.1) * 9

    is_dh = eligible & ((inn == 0) | ((inn > 0) & (estimated / inn > 3)))

    df.loc[is_dh, "Pos"] = "DH"
    return df

def filter_by_position(df, position):
    df["Pos"] = df["Pos"].astype(str).str.strip().str.upper()
    df = apply_dh_override(df)
    
    if position == "all" or "Pos" not in df.columns:
        return df
    
    position = position.upper()
    
    def player_matches(player_df):
        if position == "OF":
            of_positions = {"LF", "CF", "RF"}
            primary = player_df["Pos"].mode().iloc[0]
            return primary in of_positions
        else:
            primary = player_df["Pos"].mode().iloc[0]
            return primary == position
    
    # Group by player, check if their primary pos matches, return all their rows if so
    matched_players = (
        df.groupby("PlayerId")
        .filter(player_matches)
    )
    
    return matched_players


s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

bucket = "sports-analytics-files"

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_year(year: int) -> pd.DataFrame:
    key = f"processed/hitting_final_{year}.csv"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj["Body"])
        df["Season"] = year
        return df
    except Exception:
        return pd.DataFrame()
    
# old loading function
    """@st.cache_data(show_spinner=False, ttl=3600)
def load_final_year(year: int) -> pd.DataFrame:
    path = f"data/final/hitting_final_{year}.csv"
    try:
        df = pd.read_csv(path)
        df["Season"] = year
    
        return df
    except Exception:
        return pd.DataFrame()"""



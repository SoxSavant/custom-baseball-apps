import requests
import time

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fangraphs.com/",
    "Connection": "keep-alive"
})



def safe_get(url, **kwargs):
    for _ in range(3):
        try:
            r = SESSION.get(url, timeout=15, **kwargs)
            if r.status_code == 200:
                return r
        except:
            pass
        time.sleep(2)
    return None

requests.get = SESSION.get
requests.post = SESSION.post

import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import html
import re
import io
from pathlib import Path
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.set_page_config(page_title="Pitching Year-over-Year", layout="wide", page_icon="⚾",)

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;} 
        .viewerBadge_link__qRi_k {display: none;}
        div.ag-header-cell[col-id="ag-RowSelector"],
        div.ag-pinned-left-cols-container [col-id="ag-RowSelector"],
        div.ag-center-cols-container [col-id="ag-RowSelector"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

STATCAST_START_YEAR = 2015
STATCAST_RATE_STATS = {"Barrel%", "HardHit%", "EV"}

# ----------------------------
#  CONSTANTS
# ----------------------------

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

STAT_ALLOWLIST = [
    "WAR", "ERA", "xERA", "FIP", "xFIP", "IP", "G", "GS", "W", "L", "SV", "SO", "BB", "K/9", "BB/9",
    "HR/9", "K%", "BB%", "K-BB%", "WHIP", "ERA-", "FIP-", "Barrel%", "HardHit%", "EV",
    "O-Swing%", "Contact%", "GB%", "FB%", "CG", "ShO"
]

label_map = {
    "HardHit%": "Hard Hit%",
    "WAR":      "fWAR",
    "EV":       "Avg Exit Velo",
    "O-Swing%": "Chase%",
    "Contact%": "Whiff%",
}

# For pitchers, lower is better for these stats
lower_better = {
    "HardHit%", "Barrel%", "EV", "ERA", "xERA", "FIP", "xFIP", "BB", "HBP", "HR",
    "BB/9", "HR/9", "BABIP", "HR/FB", "BB%", "AVG", "WHIP", "ERA-", "FIP-",
    "FB%", "SIERA", "Z-Swing%", "Pull%", "LD%", "L"
}

HEADSHOT_BASES = [
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current",
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current",
]
HEADSHOT_BREF_BASES = [
    "https://content-static.baseball-reference.com/req/202406/images/headshots/{folder}/{bref_id}.jpg",
    "https://content-static.baseball-reference.com/req/202310/images/headshots/{folder}/{bref_id}.jpg",
]
HEADSHOT_CHECK_TIMEOUT = 1.0
HEADSHOT_USER_AGENT = "headshot-fetcher/1.0"
HEADSHOT_PLACEHOLDER = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0nMjQwJyBoZWlnaHQ9JzI0MCcgdmlld0JveD0nMCAwIDI0MCAyNDAnIHhtbG5zPSdodHRwOi8v"
    "d3d3LnczLm9yZy8yMDAwL3N2Zyc+CjxyZWN0IHdpZHRoPScyNDAnIGhlaWdodD0nMjQwJyBmaWxsPScjZWVmJy8+"
    "CjxjaXJjbGUgY3g9JzEyMCcgY3k9Jzk1JyByPSc1NScgZmlsbD0nI2RkZScvPgo8Y2lyY2xlIGN4PScxMjAnIGN5"
    "PSc4NScgcj0nNDInIGZpbGw9JyNmZmYnIHN0cm9rZT0nI2NjYycvPgo8cGF0aCBkPSdNMTIwIDE1MGMtMzAgMC01"
    "NSAyNS01NSA1NXMzNSAxNS41IDU1IDE1LjUgNTUtMTUuNSA1NS0xNS41LTM1LTU1LTU1LTU1eicgZmlsbD0nI2Nj"
    "YycvPgo8L3N2Zz4="
)

SUM_STATS = {"G", "GS", "W", "L", "SV", "HLD", "BS", "SO", "BB", "HR", "ER", "H", "HBP",
             "IBB", "WP", "BK", "R", "WAR", "CG", "ShO", "TBF"}
RATE_STATS = {"FIP", "xFIP", "xERA", "WHIP", "K/9", "BB/9", "HR/9", "K%", "BB%",
              "K-BB%", "Barrel%", "HardHit%", "EV", "O-Swing%", "GB%", "FB%",
              "LD%", "HR/FB", "BABIP", "SIERA", "ERA-", "FIP-"}


# ----------------------------
#  UTILITY FUNCTIONS
# ----------------------------

def normalize_statcast_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    cleaned = name.replace("\xa0", " ").strip()
    if "," in cleaned:
        last, first = cleaned.split(",", 1)
        full = f"{first.strip()} {last.strip()}"
    else:
        full = cleaned
    try:
        full = unicodedata.normalize("NFKD", full).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(full.split())


def normalize_team_code(team: str, year: int) -> str | None:
    if not team:
        return None
    team = team.upper().strip()
    if team in {"", "-", "--", "---", "- - -", "TOT"}:
        return None
    if year < 2025:
        if team in {"ATH", "OAK"}:
            return "OAK"
    else:
        if team in {"ATH", "OAK"}:
            return "ATH"
    return team


def ip_to_outs(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    if isinstance(value, str):
        match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", value)
        if not match:
            return np.nan
        try:
            v = float(match.group(0))
        except Exception:
            return np.nan
    else:
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
        outs_extra = int(round(fractional * 3))
        outs_extra = min(max(outs_extra, 0), 2)
    return innings * 3 + outs_extra


def outs_to_ip(outs: float) -> float:
    if pd.isna(outs):
        return np.nan
    total_outs = float(outs)
    innings = int(total_outs // 3)
    remainder = int(round(total_outs % 3))
    return innings + remainder / 10


def optimize_dtypes(df):
    if df.empty:
        return df
    df = df.copy()
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        if col not in {"ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9"}:
            df[col] = df[col].astype("float32")
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        if df[col].max() < 2147483647:
            df[col] = df[col].astype("int32")
    return df


def format_stat(stat: str, val, show_sign: bool = False) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat == "AGE":
        if isinstance(val, str):
            return val
        v = float(val)
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"WAR", "FWAR", "EV"}:
        v = float(val)
        formatted = f"{abs(v):.1f}" if abs(v - round(v)) >= 1e-9 else f"{int(round(abs(v)))}.0"
        if show_sign and v > 0:
            return f"+{formatted}"
        elif v < 0:
            return f"-{formatted}"
        return formatted

    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "K/9", "BB/9", "HR/9"}:
        v = float(val)
        formatted = f"{abs(v):.2f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        elif v < 0:
            return f"-{formatted}"
        return formatted

    if upper_stat == "WHIP":
        v = float(val)
        formatted = f"{abs(v):.3f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        elif v < 0:
            return f"-{formatted}"
        return formatted

    if upper_stat == "IP":
        v = float(val)
        formatted = f"{abs(v):.1f}" if abs(v - round(v)) >= 1e-9 else f"{int(round(abs(v)))}.0"
        if show_sign and v > 0:
            return f"+{formatted}"
        elif v < 0:
            return f"-{formatted}"
        return formatted

    if upper_stat in {"ERA-", "FIP-"}:
        v = int(round(float(val)))
        if show_sign and v > 0:
            return f"+{v}"
        return f"{v}"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat or "BB%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        formatted = f"{abs(v):.1f}%"
        if show_sign and v > 0:
            return f"+{formatted}"
        elif v < 0:
            return f"-{abs(v):.1f}%"
        return formatted

    v = float(val)
    formatted = f"{abs(v):.0f}" if abs(v - round(v)) < 1e-6 else f"{abs(v):.1f}"
    if show_sign and v > 0:
        return f"+{formatted}"
    elif v < 0:
        return f"-{formatted}"
    return formatted


def transform_stat_value(stat: str, raw_val):
    if stat == "Contact%":
        if pd.isna(raw_val):
            return np.nan
        try:
            contact = float(raw_val)
        except Exception:
            return np.nan
        if contact <= 1:
            contact *= 100
        return 100 - contact
    return raw_val


# ----------------------------
#  DATA LOADING
# ----------------------------

@st.cache_data(ttl=3600, max_entries=10)
def pitching_stats_cached(year: int, qual=0):
    try:
        df = pybaseball.pitching_stats(year, year, qual=qual, split_seasons=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def load_single_year(year: int, min_ip: int = 0) -> pd.DataFrame:
    """Load and process a single year of pitching stats, one row per pitcher."""
    df = pitching_stats_cached(year, qual=min_ip)
    if df is None or df.empty:
        return pd.DataFrame()

    df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")

    # Collapse multi-team pitchers to one row
    collapsed = []
    for fg_id, grp in df.groupby("IDfg"):
        raw_teams = grp["Team"].dropna().astype(str).str.strip().str.upper().tolist()
        teams = [normalize_team_code(t, year) for t in raw_teams if t not in {"TOT", "---", "--", "-", ""}]
        teams = sorted(set(t for t in teams if t))

        # For stats, prefer TOT row if it exists
        tot_row = grp[grp["Team"] == "TOT"]
        base = tot_row.iloc[0].to_dict() if not tot_row.empty else grp.iloc[0].to_dict()

        if len(teams) == 1:
            base["TeamDisplay"] = teams[0]
        elif teams:
            base["TeamDisplay"] = "2+ Teams"
        else:
            base["TeamDisplay"] = "2+ Teams"

        base["_teams_list"] = teams
        collapsed.append(base)

    df = pd.DataFrame(collapsed)

    # Convert Contact% to Whiff% in place so deltas are computed correctly
    if "Contact%" in df.columns:
        contact = pd.to_numeric(df["Contact%"], errors="coerce")
        median_val = contact.median()
        if pd.notna(median_val) and median_val <= 1:
            contact = contact * 100
        df["Contact%"] = 100 - contact

    return df


# ----------------------------
#  HEADSHOT HELPERS
# ----------------------------

@st.cache_data(show_spinner=False, tttl=3600)
def lookup_mlbam_id(full_name: str, return_bbref: bool = False):
    if not full_name or not full_name.strip():
        return (None, None) if return_bbref else None
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

    def normalize_token(tok):
        if not tok:
            return ""
        tok = tok.replace(".", "").strip()
        try:
            return unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode()
        except Exception:
            return tok

    def clean_full(val):
        try:
            val = unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
        except Exception:
            pass
        return "".join(ch for ch in val if ch.isalnum()).lower()

    def strip_suffix(tokens):
        toks = tokens.copy()
        while toks and toks[-1].lower() in suffixes:
            toks.pop()
        return toks

    parts = full_name.split()
    base_tokens = strip_suffix(parts)
    if len(base_tokens) < 2:
        return (None, None) if return_bbref else None

    first_raw = base_tokens[0]
    last_raw = " ".join(base_tokens[1:])
    target_clean = clean_full(first_raw + last_raw)

    variants = [
        (last_raw, first_raw),
        (normalize_token(last_raw), normalize_token(first_raw)),
        (normalize_token(last_raw).lower(), normalize_token(first_raw).lower()),
        (last_raw.replace(".", ""), first_raw.replace(".", "")),
    ]

    best_match_mlbam = None
    best_match_bbref = None
    first_hit_mlbam = None
    first_hit_bbref = None

    def consider_row(row):
        nonlocal first_hit_mlbam, first_hit_bbref, best_match_mlbam, best_match_bbref
        combo = clean_full(str(row.get("name_first", "")) + str(row.get("name_last", "")))
        mlbam_val = row.get("key_mlbam")
        bbref_val = row.get("key_bbref")
        if combo == target_clean:
            if pd.notna(mlbam_val):
                try:
                    best_match_mlbam = int(mlbam_val)
                except Exception:
                    pass
            if pd.notna(bbref_val):
                best_match_bbref = str(bbref_val)
        if first_hit_mlbam is None and pd.notna(mlbam_val):
            try:
                first_hit_mlbam = int(mlbam_val)
            except Exception:
                pass
        if first_hit_bbref is None and pd.notna(bbref_val):
            first_hit_bbref = str(bbref_val)

    for last, first in variants:
        try:
            lookup_df = pybaseball.playerid_lookup(last, first)
        except Exception:
            continue
        if lookup_df is None or lookup_df.empty:
            continue
        for _, row in lookup_df.iterrows():
            consider_row(row)

    mlbam_result = best_match_mlbam if best_match_mlbam is not None else first_hit_mlbam
    bbref_result = best_match_bbref if best_match_bbref is not None else first_hit_bbref
    if return_bbref:
        return mlbam_result, bbref_result
    return mlbam_result


@st.cache_data(show_spinner=False, ttl=21600)
def build_mlb_headshot(mlbam) -> str | None:
    if mlbam is None:
        return None
    mlbam_val = str(mlbam).strip()
    headers = {"User-Agent": HEADSHOT_USER_AGENT}
    for base in HEADSHOT_BASES:
        try:
            url = base.format(mlbam=mlbam_val)
            resp = requests.head(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200:
                return url
            if resp.status_code in (403, 404, 405):
                resp_get = requests.get(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, stream=True)
                if resp_get.status_code == 200:
                    return url
        except Exception:
            continue
    return HEADSHOT_BASES[0].format(mlbam=mlbam_val)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def reverse_lookup_mlbam(fg_id: int) -> int | None:
    try:
        rev = pybaseball.playerid_reverse_lookup([int(fg_id)], key_type="fangraphs")
        if rev is not None and not rev.empty:
            mlbam = rev.iloc[0].get("key_mlbam")
            if pd.notna(mlbam):
                return int(mlbam)
    except Exception:
        pass
    return None


def build_bref_headshot(bref_id: str | None) -> str | None:
    if not bref_id:
        return None
    slug = str(bref_id).strip()
    if not slug:
        return None
    for base in HEADSHOT_BREF_BASES:
        try:
            return base.format(folder=slug[0].lower(), bref_id=slug)
        except Exception:
            continue
    return None


def get_headshot_url_from_row(row: pd.Series) -> str:
    name = str(row.get("Name", "")).strip()
    for col in ["mlbam_override", "mlbamid", "mlbam_id", "mlbam", "MLBID", "MLBAMID", "key_mlbam"]:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                try:
                    headshot = build_mlb_headshot(int(val))
                    if headshot:
                        return headshot
                except Exception:
                    pass
    for col in ["playerid", "IDfg", "fg_id", "FGID"]:
        if col in row.index:
            fg = row.get(col)
            if pd.notna(fg) and str(fg).strip():
                try:
                    mlbam = reverse_lookup_mlbam(int(fg))
                    if mlbam:
                        headshot = build_mlb_headshot(mlbam)
                        if headshot:
                            return headshot
                except Exception:
                    pass
    if name:
        mlbam_fallback, bbref_fallback = lookup_mlbam_id(name, return_bbref=True)
        if mlbam_fallback:
            headshot = build_mlb_headshot(mlbam_fallback)
            if headshot:
                return headshot
        if bbref_fallback:
            bref_url = build_bref_headshot(bbref_fallback)
            if bref_url:
                return bref_url
    return HEADSHOT_PLACEHOLDER


# ----------------------------
#  CORE RISERS/FALLERS LOGIC
# ----------------------------

def load_risers_data(start_year: int, end_year: int, min_ip: int = 0,
                     team: str = "all") -> pd.DataFrame:
    """
    Load two years of pitching data, find pitchers in both with >= min_ip each year,
    compute the delta (end - start) for all stats.
    """
    df_start = load_single_year(start_year, min_ip=min_ip)
    df_end   = load_single_year(end_year,   min_ip=min_ip)

    if df_start.empty or df_end.empty:
        return pd.DataFrame()

    # Team filter: end year only
    if team != "all":
        def played_for_team(teams_list, yr):
            if not teams_list:
                return False
            return any(normalize_team_code(t, yr) == team for t in teams_list)

        if "_teams_list" in df_end.columns:
            df_end = df_end[df_end["_teams_list"].apply(lambda t: played_for_team(t, end_year))]

    df_start["IDfg"] = pd.to_numeric(df_start["IDfg"], errors="coerce")
    df_end["IDfg"]   = pd.to_numeric(df_end["IDfg"],   errors="coerce")

    # Apply min_ip filter properly — use IP column directly
    if min_ip > 0:
        if "IP" in df_start.columns:
            df_start = df_start[pd.to_numeric(df_start["IP"], errors="coerce").fillna(0) >= min_ip]
        if "IP" in df_end.columns:
            df_end = df_end[pd.to_numeric(df_end["IP"], errors="coerce").fillna(0) >= min_ip]

    common_ids = set(df_start["IDfg"].dropna()) & set(df_end["IDfg"].dropna())
    df_start = df_start[df_start["IDfg"].isin(common_ids)].set_index("IDfg")
    df_end   = df_end[df_end["IDfg"].isin(common_ids)].set_index("IDfg")

    if df_start.empty or df_end.empty:
        return pd.DataFrame()

    skip_meta = {"Name", "Team", "TeamDisplay", "_teams_list", "mlbam", "MLBID",
                 "key_mlbam", "mlbam_id", "Season"}
    numeric_cols = [
        c for c in df_end.columns
        if c not in skip_meta
        and pd.api.types.is_numeric_dtype(df_end[c])
        and c in df_start.columns
    ]

    result_rows = []
    for fg_id in common_ids:
        if fg_id not in df_start.index or fg_id not in df_end.index:
            continue
        row_s = df_start.loc[fg_id]
        row_e = df_end.loc[fg_id]

        record = {
            "IDfg": fg_id,
            "Name": row_e.get("Name", row_s.get("Name", "")),
            "TeamDisplay": row_e.get("TeamDisplay", ""),
        }

        # Carry over MLBAM for headshots
        for hcol in ["mlbam", "MLBID", "key_mlbam", "mlbam_id"]:
            if hcol in row_e.index and pd.notna(row_e.get(hcol)):
                record[hcol] = row_e[hcol]
            elif hcol in row_s.index and pd.notna(row_s.get(hcol)):
                record[hcol] = row_s[hcol]

        for col in numeric_cols:
            try:
                s_val = pd.to_numeric(row_s[col] if col in row_s.index else np.nan, errors="coerce")
                e_val = pd.to_numeric(row_e[col] if col in row_e.index else np.nan, errors="coerce")
                record[f"{col}_start"] = s_val
                record[f"{col}_end"]   = e_val
                record[col] = e_val - s_val  # delta
            except Exception:
                record[col] = np.nan

        record["IP_start"] = pd.to_numeric(row_s.get("IP", np.nan), errors="coerce")
        record["IP_end"]   = pd.to_numeric(row_e.get("IP", np.nan), errors="coerce")

        result_rows.append(record)

    if not result_rows:
        return pd.DataFrame()

    return pd.DataFrame(result_rows)


# ----------------------------
#  PAGE HEADER
# ----------------------------
title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Year-over-Year Pitcher Risers & Fallers")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

current_year = date.today().year

# ----------------------------
#  SESSION STATE DEFAULTS
# ----------------------------
for key, default in [
    ("pr_start_year",     2024),
    ("pr_end_year",       2025),
    ("pr_stat",           "ERA"),
    ("pr_min_ip",         100),
    ("pr_team",           "all"),
    ("pr_show_fallers",   False),
    ("pr_show_min_ip",    True),
    ("pr_show_player_ip", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(
    """
    <style>
        .stSelectbox div[data-baseweb="select"],
        .stNumberInput > div { max-width: 200px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
#  CONTROLS
# ----------------------------
stat = st.selectbox(
    "Stat",
    STAT_ALLOWLIST,
    key="pr_stat",
    format_func=lambda x: label_map.get(x, x),
)

col1, col2 = st.columns([0.5, 2])

with col1:
    st.number_input("Start Year", min_value=1900, max_value=current_year, key="pr_start_year")
    st.number_input("End Year",   min_value=1900, max_value=current_year, key="pr_end_year")

    start_year = st.session_state["pr_start_year"]
    end_year   = st.session_state["pr_end_year"]
    if end_year <= start_year:
        st.warning("End Year must be greater than Start Year.")

    st.number_input("Min IP (each year)", min_value=0, max_value=5000, key="pr_min_ip")

    st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key="pr_team",
        help="Filters by team in the end year only",
    )

    st.checkbox("Show Fallers",       key="pr_show_fallers")
    st.checkbox("Show min IP",        key="pr_show_min_ip")
    st.checkbox("Show player IP",     key="pr_show_player_ip")

# Resolved values
min_ip_val   = int(st.session_state.get("pr_min_ip", 0))
team_val     = st.session_state.get("pr_team", "all")
show_fallers = st.session_state.get("pr_show_fallers", False)

# ----------------------------
#  LOAD DATA
# ----------------------------
if end_year > start_year:
    with st.spinner("Loading data..."):
        df = load_risers_data(start_year, end_year, min_ip_val, team_val)
else:
    df = pd.DataFrame()

# ----------------------------
#  SORT & FILTER
# ----------------------------
if not df.empty and stat in df.columns:
    stat_is_lower_better = stat in lower_better

    if stat_is_lower_better:
        ascending = not show_fallers
    else:
        ascending = show_fallers

    df = df.sort_values(by=stat, ascending=ascending)

    # Only show improvements for risers, only declines for fallers
    numeric_delta = pd.to_numeric(df[stat], errors="coerce")
    if stat_is_lower_better:
        df = df[numeric_delta < 0] if not show_fallers else df[numeric_delta > 0]
    else:
        df = df[numeric_delta > 0] if not show_fallers else df[numeric_delta < 0]

    df = df.head(10)
elif not df.empty:
    st.error(f"Column '{stat}' not found. Available columns: {', '.join(df.columns)}")
    df = pd.DataFrame()

# ----------------------------
#  BUILD CARDS
# ----------------------------
cards = []
for _, row in df.iterrows():
    name      = row.get("Name", "")
    team      = row.get("TeamDisplay", "")
    raw_delta = row.get(stat, np.nan)

    display_delta = raw_delta
    transformed_delta = display_delta
    if stat == "Contact%" and pd.notna(transformed_delta):
        transformed_delta = float(transformed_delta) / 100
    is_positive = pd.notna(transformed_delta) and float(transformed_delta) > 0
    display_val = format_stat(stat, transformed_delta, show_sign=is_positive)

    # End-year value for context
    end_val_raw = row.get(f"{stat}_end", np.nan)
    if stat == "Contact%" and pd.notna(end_val_raw):
        end_val_raw = float(end_val_raw) / 100
    end_display = format_stat(stat, end_val_raw) if pd.notna(end_val_raw) else ""
    stat_label = label_map.get(stat, stat)

    src_row = row
    try:
        pos = list(df.index).index(row.name)
        key = f"pr_mlbam_override_{pos}"
        override_val = st.session_state.get(key, "")
        if override_val and str(override_val).strip():
            try:
                src_row = row.copy()
                src_row["mlbam_override"] = int(str(override_val).strip())
            except Exception:
                pass
    except Exception:
        pass

    ip_start = row.get("IP_start", np.nan)
    ip_end   = row.get("IP_end",   np.nan)
    player_ip_display = ""
    if st.session_state.get("pr_show_player_ip", False):
        parts_ip = []
        if pd.notna(ip_start):
            parts_ip.append(format_stat("IP", ip_start))
        if pd.notna(ip_end):
            parts_ip.append(format_stat("IP", ip_end))
        if parts_ip:
            player_ip_display = f'<div class="player-ip">{" → ".join(parts_ip)} IP</div>'

    src = get_headshot_url_from_row(src_row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(str(name))}"/>'

    # Color: green = improvement, red = decline
    if stat in lower_better:
        is_improvement = pd.notna(transformed_delta) and float(transformed_delta) < 0
    else:
        is_improvement = is_positive
    delta_class = "stat-positive" if is_improvement else "stat-negative"

    end_context = f'<div class="player-endval">{stat_label}: {end_display}</div>' if end_display else ""

    card_html = f'''
    <div class="player-card">
      {img_html}
      <div class="player-name">{name}</div>
      <div class="player-team">{team}</div>
      <div class="player-stat {delta_class}">{display_val}</div>
      {end_context}
      {player_ip_display}
    </div>
    '''
    cards.append(card_html)

# ----------------------------
#  BUILD TITLE
# ----------------------------
title_stat_label = label_map.get(stat, stat)
team_prefix  = f"{TEAM_OPTIONS.get(team_val, '')} " if team_val != "all" else ""
riser_label  = "Fallers" if show_fallers else "Risers"

title = f"Top {team_prefix}{title_stat_label} {riser_label}: {int(start_year)} → {int(end_year)}"

min_ip_subtitle = ""
if st.session_state.get("pr_show_min_ip", False):
    min_ip_subtitle = f'<div class="leaderboard-subtitle">Min {min_ip_val} IP each year</div>'

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{title}</div>
    {min_ip_subtitle}
    <div class="players-grid">
        {''.join(cards) if cards else '<div style="padding:2rem;color:#999;">No data found. Try adjusting your filters or years.</div>'}
    </div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p>Data: FanGraphs</p>
    </div>
</div>
"""

full_html = f"""
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800&display=swap" rel="stylesheet">
<meta charset="utf-8" />
<style>
.leaderboard-card {{
    background: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 12px;
    padding: 2rem 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto;
    width: 100%;
    max-width: 900px;
    box-sizing: border-box;
    font-family: "Source Sans Pro", sans-serif;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.2rem;
    margin-bottom: 1rem;
    text-align: center;
}}
.leaderboard-subtitle {{
    text-align: center;
    color: #888;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    margin-top: -0.5rem;
}}
.players-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2rem 1rem;
}}
.player-card {{
    flex: 0 0 145px;
    width: 145px;
    text-align: center;
}}
.player-card img {{
    width: 145px;
    height: 145px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    background: #f6f6f6;
}}
.player-name {{
    font-weight: 800;
    margin-top: 0.35rem;
    font-size: 1.1rem;
}}
.player-team {{
    color: #666;
    font-size: 0.85rem;
}}
.player-stat {{
    font-weight: 900;
    font-size: 1.5rem;
    margin-top: 0.25rem;
}}
.stat-positive {{
    color: #1a7a3c;
}}
.stat-negative {{
    color: #c0392b;
}}
.player-endval {{
    color: #888;
    font-size: .9rem;
    margin-top: 0.1rem;
}}
.player-ip {{
    color: #666;
    font-size: 0.9rem;
    margin-top: 0.1rem;
}}
html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    width: 100%;
}}
.footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.5rem;
}}
.footer p {{
    margin: 0;
    font-size: 0.9rem;
    color: #666;
    flex: 1;
    text-align: center;
}}
.footer p:first-child {{ text-align: left; }}
.footer p:last-child  {{ text-align: right; }}
</style>
</head>
<body>
{grid_html}
</body>
</html>
"""

with col2:
    components.html(full_html, height=850)

# ----------------------------
#  MLBAM OVERRIDES
# ----------------------------
if not df.empty:
    st.markdown("---")
    st.write("Manual MLBAM overrides (enter MLBAM id to fix headshot)")

    for row_offset in range(0, 10, 5):
        cols_row = st.columns(5)
        for col_idx in range(5):
            player_idx = row_offset + col_idx
            if player_idx >= len(df):
                break
            idx = df.index[player_idx]
            row = df.loc[idx]
            with cols_row[col_idx]:
                key = f"pr_mlbam_override_{player_idx}"
                default_val = ""
                if "mlbam_override" in df.columns and pd.notna(row.get("mlbam_override")):
                    try:
                        default_val = str(int(row["mlbam_override"]))
                    except Exception:
                        default_val = str(row.get("mlbam_override", ""))
                user_val = st.text_input(f"Player {player_idx+1} MLBAM", value=default_val, key=key)
                try:
                    df.at[idx, "mlbam_override"] = int(str(user_val).strip()) if user_val and str(user_val).strip() else np.nan
                except Exception:
                    df.at[idx, "mlbam_override"] = np.nan

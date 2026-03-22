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
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.set_page_config(page_title="Pitcher Stat Filter Leaderboard", layout="wide", page_icon="⚾")

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;}
        .viewerBadge_link__qRi_k {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Stats available for combo filters
COMBO_STATS = [
    "WAR", "ERA", "xERA", "FIP", "xFIP", "WHIP", "ERA-", "FIP-", "SIERA",
    "IP", "G", "GS", "W", "L", "SV", "HLD", "SO", "BB",
    "K/9", "BB/9", "HR/9", "K%", "BB%", "K-BB%",
    "Barrel%", "HardHit%", "EV",
    "O-Swing%", "Contact%", "GB%", "FB%", "LD%", "HR/FB",
    "BABIP", "Age", "WPA", "Clutch", "CG", "ShO",
]

# Stats where lower = better
LOWER_BETTER = {
    "ERA", "xERA", "FIP", "xFIP", "WHIP", "ERA-", "FIP-", "SIERA",
    "BB", "BB/9", "HR/9", "BB%", "HardHit%", "Barrel%", "EV",
    "O-Swing%", "Contact%", "FB%", "HR/FB", "BABIP", "L",
}

label_map = {
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
    "EV": "Avg Exit Velo",
    "Contact%": "Whiff%",
    "O-Swing%": "Chase%",
}

# Default thresholds per stat
STAT_DEFAULTS = {
    "WAR": 3.0, "ERA": 3.00, "xERA": 3.00, "FIP": 3.00, "xFIP": 3.00,
    "WHIP": 1.10, "ERA-": 80.0, "FIP-": 80.0, "SIERA": 3.50,
    "IP": 162.0, "G": 30.0, "GS": 25.0, "W": 12.0, "L": 10.0,
    "SV": 20.0, "HLD": 15.0, "SO": 180.0, "BB": 50.0,
    "K/9": 10.0, "BB/9": 2.5, "HR/9": 1.0,
    "K%": 25.0, "BB%": 7.0, "K-BB%": 18.0,
    "Barrel%": 6.0, "HardHit%": 35.0, "EV": 88.0,
    "O-Swing%": 32.0, "Contact%": 20.0,
    "GB%": 50.0, "FB%": 35.0, "LD%": 20.0, "HR/FB": 10.0,
    "BABIP": 0.280, "Age": 28.0, "WPA": 2.0, "Clutch": 1.0,
    "CG": 1.0, "ShO": 1.0,
}

HEADSHOT_BASES = [
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current",
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current",
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
MAX_DISPLAY = 30

# ----------------------------
#  HELPERS
# ----------------------------

def update_stat_default(i):
    stat = st.session_state[f"pc_stat_{i}"]
    st.session_state[f"pc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"pc_op_{i}"] = "<=" if stat in LOWER_BETTER else ">="


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


def normalize_team_code(team: str, year: int):
    if not team:
        return None
    team = team.upper().strip()
    if team in {"", "-", "--", "---", "TOT"}:
        return None
    if year < 2025:
        if team in {"ATH", "OAK"}:
            return "OAK"
    else:
        if team in {"ATH", "OAK"}:
            return "ATH"
    return team


def is_junk_team(t: str) -> bool:
    return t == "TOT" or t.replace(" ", "").replace("-", "") == ""


def ip_to_outs(value) -> float:
    """Convert FanGraphs IP (e.g. 6.2 = 6 innings + 2 outs) to total outs."""
    try:
        v = float(value)
    except Exception:
        return np.nan
    innings = int(np.floor(v))
    frac = v - innings
    if abs(frac - 0.1) < 0.05:
        extra = 1
    elif abs(frac - 0.2) < 0.05:
        extra = 2
    else:
        extra = int(round(frac * 3))
        extra = min(max(extra, 0), 2)
    return innings * 3 + extra


def outs_to_ip(outs: float) -> float:
    if pd.isna(outs):
        return np.nan
    innings = int(outs // 3)
    remainder = int(round(outs % 3))
    return innings + remainder / 10


def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper = stat.upper()
    if upper in {"ERA", "FIP", "XFIP", "XERA", "K/9", "BB/9", "HR/9", "SIERA"}:
        return f"{float(val):.2f}"
    if upper == "WHIP":
        return f"{float(val):.3f}"
    if upper == "IP":
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    if upper in {"ERA-", "FIP-"}:
        return f"{int(round(float(val)))}"
    if upper in {"WAR", "EV", "AVG EXIT VELO"}:
        v = float(val)
        return f"{v:.1f}" if abs(v - round(v)) >= 1e-9 else f"{int(round(v))}.0"
    if upper in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"
    if upper == "BABIP":
        return f"{float(val):.3f}".lstrip("0") or ".000"
    if "%" in stat or any(x in stat for x in ["Barrel", "Hard", "K%", "BB%", "Swing", "Whiff"]):
        v = float(val)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"
    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def format_threshold(stat: str, val: float, op: str) -> str:
    label = label_map.get(stat, stat)
    formatted = format_stat(stat, val)
    numeric = formatted.rstrip("%")
    if op == ">=":
        return f"{numeric}+ {label}"
    else:
        return f"≤ {numeric} {label}"


# ----------------------------
#  DATA LOADING
# ----------------------------

@st.cache_data(ttl=600, max_entries=10)
def pitching_stats_cached(year: int, qual: int = 0):
    try:
        df = pybaseball.pitching_stats(year, year, qual=qual, split_seasons=False)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, max_entries=10)
def load_year(year: int, min_ip: int = 0) -> pd.DataFrame:
    df = pitching_stats_cached(year, qual=min_ip)
    if df is None or df.empty:
        return pd.DataFrame()

    df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")

    # Resolve TOT rows — keep individual team rows, fall back to TOT only if no individual rows exist
    tot_ids = set(df.loc[df["Team"] == "TOT", "IDfg"])
    has_ind = set(df.loc[df["Team"] != "TOT", "IDfg"])
    non_tot = df[df["Team"] != "TOT"]
    tot_fb  = df[(df["Team"] == "TOT") & (df["IDfg"].isin(tot_ids - has_ind))]
    df = pd.concat([non_tot, tot_fb], ignore_index=True)
    df = df[df["Team"].notna()]

    # Collapse multi-team pitchers to one row, carrying TOT stats
    collapsed = []
    for fg_id, grp in df.groupby("IDfg"):
        raw_teams = grp["Team"].dropna().astype(str).str.strip().str.upper().tolist()
        teams = sorted(set(
            t for t in [normalize_team_code(x, year) for x in raw_teams if not is_junk_team(x)]
            if t
        ))
        tot_row = grp[grp["Team"] == "TOT"]
        base = tot_row.iloc[0].to_dict() if not tot_row.empty else grp.iloc[0].to_dict()
        if len(teams) == 1:
            base["TeamDisplay"] = teams[0]
        elif len(teams) > 1:
            base["TeamDisplay"] = "2+ Teams"
        else:
            fallback = next((x for x in raw_teams if not is_junk_team(x)), "")
            base["TeamDisplay"] = fallback if fallback else "2+ Teams"
        base["_teams_list"] = teams
        collapsed.append(base)

    df = pd.DataFrame(collapsed)
    return df


# ----------------------------
#  HEADSHOT FUNCTIONS
# ----------------------------

@st.cache_data(show_spinner=False)
def lookup_mlbam_id(full_name: str, return_bbref: bool = False):
    if not full_name or not full_name.strip():
        return (None, None) if return_bbref else None
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

    def normalize_token(tok):
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

    parts = full_name.split()
    while parts and parts[-1].lower() in suffixes:
        parts.pop()
    if len(parts) < 2:
        return (None, None) if return_bbref else None

    first_raw = parts[0]
    last_raw = " ".join(parts[1:])
    target_clean = clean_full(first_raw + last_raw)

    variants = [
        (last_raw, first_raw),
        (normalize_token(last_raw), normalize_token(first_raw)),
        (normalize_token(last_raw).lower(), normalize_token(first_raw).lower()),
        (last_raw.replace(".", ""), first_raw.replace(".", "")),
    ]

    best_mlbam = None; best_bbref = None; first_mlbam = None; first_bbref = None

    def consider(row):
        nonlocal best_mlbam, best_bbref, first_mlbam, first_bbref
        combo = clean_full(str(row.get("name_first", "")) + str(row.get("name_last", "")))
        mv = row.get("key_mlbam"); bv = row.get("key_bbref")
        if combo == target_clean:
            if pd.notna(mv):
                try: best_mlbam = int(mv)
                except: pass
            if pd.notna(bv):
                try: best_bbref = str(bv)
                except: pass
        if first_mlbam is None and pd.notna(mv):
            try: first_mlbam = int(mv)
            except: pass
        if first_bbref is None and pd.notna(bv):
            try: first_bbref = str(bv)
            except: pass

    for last, first in variants:
        try:
            ldf = pybaseball.playerid_lookup(last, first)
            if ldf is not None and not ldf.empty:
                for _, r in ldf.iterrows(): consider(r)
        except: continue

    mlbam_res = best_mlbam if best_mlbam is not None else first_mlbam
    bbref_res = best_bbref if best_bbref is not None else first_bbref
    return (mlbam_res, bbref_res) if return_bbref else mlbam_res


@st.cache_data(show_spinner=False, ttl=21600)
def build_mlb_headshot(mlbam) -> str | None:
    if mlbam is None: return None
    mlbam_val = str(mlbam).strip()
    headers = {"User-Agent": HEADSHOT_USER_AGENT}
    for base in HEADSHOT_BASES:
        try:
            url = base.format(mlbam=mlbam_val)
            resp = requests.head(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200: return url
            if resp.status_code in (403, 404, 405):
                r2 = requests.get(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, stream=True)
                if r2.status_code == 200: return url
        except: continue
    return HEADSHOT_BASES[0].format(mlbam=mlbam_val)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def reverse_lookup_mlbam(fg_id: int) -> int | None:
    try:
        rev = pybaseball.playerid_reverse_lookup([int(fg_id)], key_type="fangraphs")
        if rev is not None and not rev.empty:
            v = rev.iloc[0].get("key_mlbam")
            if pd.notna(v): return int(v)
    except: pass
    return None


def get_headshot_url_from_row(row: pd.Series) -> str:
    name = str(row.get("Name", "")).strip()
    for col in ["mlbam_override", "mlbamid", "mlbam_id", "mlbam", "MLBID", "MLBAMID", "key_mlbam"]:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                try:
                    h = build_mlb_headshot(int(val))
                    if h: return h
                except: pass
    for col in ["playerid", "IDfg", "fg_id", "FGID"]:
        if col in row.index:
            fg = row.get(col)
            if pd.notna(fg) and str(fg).strip():
                try:
                    mlbam = reverse_lookup_mlbam(int(fg))
                    if mlbam:
                        h = build_mlb_headshot(mlbam)
                        if h: return h
                except: pass
    if name:
        mlbam_fb, _ = lookup_mlbam_id(name, return_bbref=True)
        if mlbam_fb:
            h = build_mlb_headshot(mlbam_fb)
            if h: return h
    return HEADSHOT_PLACEHOLDER


# ----------------------------
#  PAGE HEADER
# ----------------------------
title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Pitcher Stat Filter Leaderboard")
with meta_col:
    st.markdown(
        '<div style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )

current_year = date.today().year

# ----------------------------
#  SESSION STATE
# ----------------------------
for key, default in [
    ("pc_year",    2025),
    ("pc_min_ip",  100),
    ("pc_team",    "all"),
    ("pc_show_ip", False),
    ("pc_show_min_ip", True),
    ("pc_top10",   False),
    ("pc_val_0", 3.00),
    ("pc_val_1", 3.00),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
    <style>
        .stSelectbox div[data-baseweb="select"],
        .stNumberInput > div { max-width: 200px; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
#  CONTROLS
# ----------------------------
col1, col2 = st.columns([0.5, 2])

with col1:
    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=1, horizontal=True, key="pc_num_stats")
    st.number_input("Year", min_value=1900, max_value=current_year, key="pc_year")
    st.number_input("Min IP", min_value=0, max_value=5000, key="pc_min_ip")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        stat_key = f"pc_stat_{i}"
        op_key   = f"pc_op_{i}"
        val_key  = f"pc_val_{i}"

        default_stat = "ERA" if i == 0 else "FIP" if i == 1 else COMBO_STATS[0]
        default_index = COMBO_STATS.index(default_stat)

        new_stat = st.selectbox(
            f"Stat {i+1}",
            COMBO_STATS,
            key=stat_key,
            index = default_index,
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed",
            on_change=update_stat_default,
            args=(i,),
        )
        default_index = 1 if i == 0 or i == 1 else 0
        op_col, val_col = st.columns([1, 2])
        with op_col:
            st.selectbox("Op", [">=", "<="], index = default_index, key=op_key, label_visibility="collapsed")
        with val_col:
            # Three tiers: rate stats (ERA/FIP/WHIP etc) need 2-3dp, % stats need 1dp, counting are integers
            RATE_STATS_3DP = {"WHIP", "BABIP"}
            RATE_STATS_2DP = {"ERA", "xERA", "FIP", "xFIP", "SIERA", "K/9", "BB/9", "HR/9", "HR/FB"}
            if new_stat in RATE_STATS_3DP:
                step = 0.001
                fmt  = "%.3f"
            elif new_stat in RATE_STATS_2DP:
                step = 0.01
                fmt  = "%.2f"
            elif "%" in new_stat or new_stat in {"EV"}:
                step = 0.1
                fmt  = "%.1f"
            else:
                step = 1.0
                fmt  = "%.0f"
            st.number_input(
                f"Value {i+1}", step=step,
                key=val_key, label_visibility="collapsed",
                format=fmt,
            )

    st.selectbox("Team", options=list(TEAM_OPTIONS.keys()),
                 format_func=lambda x: TEAM_OPTIONS[x], key="pc_team")
    st.checkbox("Show player IP",  key="pc_show_ip")
    st.checkbox("Show min IP",     key="pc_show_min_ip")
    st.checkbox("Only display top 10", key="pc_top10")

# ----------------------------
#  LOAD & FILTER DATA
# ----------------------------
year_val   = int(st.session_state["pc_year"])
min_ip_val = int(st.session_state["pc_min_ip"])
team_val   = st.session_state["pc_team"]

with st.spinner("Loading data..."):
    df = load_year(year_val, min_ip=min_ip_val)

# Team filter
if team_val != "all" and not df.empty and "_teams_list" in df.columns:
    df = df[df["_teams_list"].apply(
        lambda tl: any(normalize_team_code(t, year_val) == team_val for t in (tl or []))
    )]

# Build active filters
active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"pc_stat_{i}")
    op   = st.session_state.get(f"pc_op_{i}", ">=")
    val  = st.session_state.get(f"pc_val_{i}", 0.0)
    if stat:
        active_filters.append((stat, op, float(val)))

# Apply filters
PCT_STATS = {
    "K%", "BB%", "K-BB%", "O-Swing%", "Contact%",
    "Barrel%", "HardHit%", "GB%", "FB%", "LD%", "HR/FB",
}

total_qualified = 0
if not df.empty:
    mask = pd.Series([True] * len(df), index=df.index)
    for stat, op, val in active_filters:
        if stat not in df.columns:
            continue
        col_vals = pd.to_numeric(df[stat], errors="coerce")
        compare_val = val
        if stat in PCT_STATS:
            median_col = col_vals.median()
            if pd.notna(median_col) and median_col <= 1:
                if val > 1:
                    compare_val = val / 100
            # Contact% stored as contact rate; user enters Whiff% threshold
            if stat == "Contact%":
                contact_threshold = 1 - (compare_val if compare_val <= 1 else compare_val / 100)
                if op == ">=":
                    mask = mask & (col_vals <= contact_threshold)
                else:
                    mask = mask & (col_vals >= contact_threshold)
                continue
        if op == ">=":
            mask = mask & (col_vals >= compare_val)
        else:
            mask = mask & (col_vals <= compare_val)

    df = df[mask]
    total_qualified = len(df)

    # Sort by first stat
    if active_filters:
        sort_stat, sort_op, _ = active_filters[0]
        if sort_stat in df.columns:
            asc = sort_op == "<="
            df = df.sort_values(sort_stat, ascending=asc)

    display_limit = 10 if st.session_state.get("pc_top10") and total_qualified > 10 else MAX_DISPLAY
    if total_qualified > display_limit:
        df = df.head(display_limit)

# ----------------------------
#  BUILD CARDS
# ----------------------------
cards = []
for card_pos, (_, row) in enumerate(df.iterrows()):
    name = row.get("Name", "")
    team = row.get("TeamDisplay", "")

    stat_lines = []
    for stat, op, threshold in active_filters:
        val = row.get(stat, np.nan)
        if pd.isna(val):
            continue
        # Contact% is stored as contact rate but displayed as whiff%
        if stat == "Contact%":
            try:
                contact = float(val)
                if contact <= 1:
                    contact *= 100
                display = f"{100 - contact:.1f}%"
            except Exception:
                display = ""
        else:
            display = format_stat(stat, val)
        lbl = label_map.get(stat, stat)
        stat_lines.append(f'<span class="stat-label">{lbl}:</span> <span class="stat-value">{display}</span>')

    ip_val = row.get("IP", np.nan)
    ip_display = (
        f'<div class="player-ip">{format_stat("IP", ip_val)} IP</div>'
        if st.session_state.get("pc_show_ip") and pd.notna(ip_val) else ""
    )

    src_row = row
    try:
        ov = st.session_state.get(f"pc_mlbam_override_{card_pos}", "")
        if ov and str(ov).strip():
            src_row = row.copy()
            src_row["mlbam_override"] = int(str(ov).strip())
    except: pass

    src = get_headshot_url_from_row(src_row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(str(name))}" width="155" height="155" style="object-fit:cover;border-radius:6px;border:1px solid #e0e0e0;background:#f6f6f6;display:block;"/>'

    cards.append(f'''
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(str(name))}</div>
      <div class="player-team">{html.escape(str(team))}</div>
      {'<div class="player-stat-line">' + " | ".join(stat_lines) + "</div>" if stat_lines else ""}
      {ip_display}
    </div>''')

# ----------------------------
#  TITLE
# ----------------------------
filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str = ", ".join(filter_parts)
team_suffix = f"({team_val}) " if team_val != "all" else ""
title = f"{filter_str} in {year_val} {team_suffix}".strip()

min_ip_subtitle = ""
if st.session_state.get("pc_show_min_ip", False):
    min_ip_subtitle = f'<div class="leaderboard-subtitle">Min {min_ip_val} IP</div>'

overflow_note = ""
display_limit = 10 if st.session_state.get("pc_top10") and total_qualified > 10 else MAX_DISPLAY
if total_qualified > display_limit:
    overflow_note = f'<div class="overflow-note">Showing top {display_limit} of {total_qualified} qualifying pitchers</div>'

if not cards:
    body = '<div style="padding:2rem;color:#999;text-align:center;">No pitchers matched all filters. Try adjusting your thresholds.</div>'
else:
    body = "".join(cards)

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_ip_subtitle}
    {overflow_note}
    <div class="players-grid">{body}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p>Data: FanGraphs</p>
    </div>
</div>
"""

card_count = len(cards)
est_rows   = max(1, (card_count + 4) // 5)
est_height = 120 + est_rows * 280 + 80

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style>
html, body {{ background: transparent; font-family: "Source Sans Pro", sans-serif; margin: 0; padding: 0; }}
.leaderboard-card {{
    background: #fff;
    border: 1px solid #d0d0d0;
    border-radius: 12px;
    padding: 3rem 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto;
    width: 100%;
    max-width: 900px;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.25rem;
    margin-bottom: 1.2rem;
    text-align: center;
    line-height: 1.2;
}}
.leaderboard-subtitle {{
    text-align: center;
    color: #888;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    margin-top: -0.5rem;
}}
.overflow-note {{
    text-align: center;
    color: #888;
    font-size: 0.95rem;
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
    flex: 0 0 155px;
    width: 155px;
    text-align: center;
}}
.player-card img {{
    width: 155px;
    height: 155px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    background: #f6f6f6;
}}
.player-name {{
    font-weight: 800;
    font-size: 1rem;
    margin-top: 0.35rem;
    line-height: 1.2;
}}
.player-team {{
    color: #666;
    font-size: 0.8rem;
    margin-bottom: 0.25rem;
}}
.player-stat-line {{
    text-align: center;
    font-size: 0.95rem;
    margin-top: 0.15rem;
}}
.stat-label  {{ color: #888; font-size: 0.85rem; }}
.stat-value  {{ font-weight: 800; font-size: 0.95rem; color: #1a1a1a; }}
.player-ip   {{ color: #aaa; font-size: 0.8rem; margin-top: 0.1rem; }}
.footer {{
    display: flex;
    justify-content: space-between;
    margin-top: 1.5rem;
    padding: 0 4rem;
}}
.footer p {{ margin: 0; font-size: 1rem; color: #888; flex: 1; text-align: center; }}
.footer p:first-child {{ text-align: left; }}
.footer p:last-child  {{ text-align: right; }}
</style>
</head>
<body>
{grid_html}
</body>
</html>"""

with col2:
    components.html(full_html, height=est_height, scrolling=True)

# ----------------------------
#  MLBAM OVERRIDES
# ----------------------------
if not df.empty:
    st.markdown("---")
    st.write("Manual MLBAM overrides (enter MLBAM id to fix headshot)")
    n = len(df)
    for row_offset in range(0, min(n, MAX_DISPLAY), 5):
        cols_row = st.columns(5)
        for col_idx in range(5):
            player_idx = row_offset + col_idx
            if player_idx >= n: break
            idx = df.index[player_idx]
            row = df.loc[idx]
            with cols_row[col_idx]:
                key = f"pc_mlbam_override_{player_idx}"
                default_val = ""
                if "mlbam_override" in df.columns and pd.notna(row.get("mlbam_override")):
                    try: default_val = str(int(row["mlbam_override"]))
                    except: pass
                user_val = st.text_input(f"Player {player_idx+1} MLBAM", value=default_val, key=key)
                try:
                    df.at[idx, "mlbam_override"] = int(str(user_val).strip()) if user_val and str(user_val).strip() else np.nan
                except:
                    df.at[idx, "mlbam_override"] = np.nan
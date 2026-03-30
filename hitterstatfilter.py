import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import unicodedata
import html
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Hitter Stat Filter Leaderboard", layout="wide", page_icon="⚾")

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;}
        .viewerBadge_link__qRi_k {display: none;}
        .stSelectbox div[data-baseweb="select"],
        .stNumberInput > div { max-width: 200px; }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Hitter Stat Filter Leaderboard")
with meta_col:
    st.markdown(
        '<div style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

TEAM_ALIASES = {"ATH": "OAK", "ATH/OAK": "OAK", "OAK/ATH": "OAK"}
LOCAL_BWAR_FILE = Path(__file__).with_name("warhitters2025.txt")
HEADSHOT_BASE = "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current"
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

STAT_ALLOWLIST = [
    "fWAR", "bWAR", "Off", "Def", "BsR",  "Barrel%", "HardHit%", "EV",  "Chase%", "Whiff%",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "TB", "H", "1B", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "WPA", "Clutch",
    "FRV", "OAA", "DRS", "FRM"
]

lower_better = {"K%", "O-Swing%", "SO", "Chase%"}

label_map = {
    "HardHit%": "Hard Hit%",
    "EV": "Avg Exit Velo",
    
}

STAT_DEFAULTS = {
    "HR": 30, "SB": 30, "RBI": 100, "R": 100, "H": 150,
    "fWAR": 4.0, "bWAR": 4.0, "wRC+": 130, "wOBA": 0.370, "OPS": 0.900,
    "xwOBA": 0.370, "xBA": 0.280, "xSLG": 0.480,
    "AVG": 0.300, "OBP": 0.370, "SLG": 0.500, "ISO": 0.200,
    "K%": 20.0, "BB%": 10.0, "Barrel%": 12.0, "HardHit%": 45.0,
    "EV": 92.0, "BB": 60, "IBB": 10, "SO": 100, "PA": 502, "AB": 450,
    "2B": 30, "1B": 100, "3B": 5, "XBH": 50, "TB": 250, "G": 140,
    "Age": 30, "Clutch": 1.0, "FRV": 10, "OAA": 10, "DRS": 10,
    "Chase%": 25.0, "Whiff%": 20.0,
}

PCT_STATS = {
    "K%", "BB%", "K-BB%", "Chase%", "Whiff%",
    "Barrel%", "HardHit%",
}

POSITION_OPTIONS = {
    "all": "All Positions", "C": "C", "1B": "1B", "2B": "2B", "3B": "3B",
    "SS": "SS", "LF": "LF", "CF": "CF", "RF": "RF", "OF": "OF", "DH": "DH",
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

current_year = date.today().year


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_bwar() -> pd.DataFrame:
    if not LOCAL_BWAR_FILE.exists():
        return pd.DataFrame()
    try:
        # 1. Read the raw data
        df = pd.read_csv(LOCAL_BWAR_FILE)
    except Exception:
        return pd.DataFrame()
    
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    
    # 2. Standardize IDs and Years
    df["MLBAMID"] = pd.to_numeric(df.get("mlb_ID"), errors="coerce")
    df["year_ID"] = pd.to_numeric(df.get("year_ID"), errors="coerce")
    df["bWAR"] = pd.to_numeric(df.get("WAR"), errors="coerce")
    
    # 3. Clean up missing values before aggregating
    df = df.dropna(subset=["MLBAMID", "year_ID", "bWAR"])

    # 4. THE FIX: Group by ID and Year, then SUM the WAR
    # This combines traded players (e.g. 0.5 WAR + 1.2 WAR) into one 1.7 WAR row
    df = df.groupby(["MLBAMID", "year_ID"], as_index=False)["bWAR"].sum()

    return df[["MLBAMID", "year_ID", "bWAR"]]


def normalize_team(team: str) -> str:
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)


def get_team_display(team_value: str) -> str:
    t = str(team_value).strip()
    if t == "- - -":
        return "2+ Teams"
    return normalize_team(t)


def get_headshot(row: pd.Series) -> str:
    for col in ["MLBAMID", "mlbamid", "mlbam_id", "MLBID"]:
        val = row.get(col)
        if val is not None and pd.notna(val):
            try:
                return HEADSHOT_BASE.format(mlbam=int(val))
            except Exception:
                pass
    return HEADSHOT_PLACEHOLDER


def apply_dh_override(df: pd.DataFrame) -> pd.DataFrame:
    if "Pos" not in df.columns or "PA" not in df.columns or "Inn" not in df.columns:
        return df
    df = df.copy()
    pa  = pd.to_numeric(df["PA"],  errors="coerce").fillna(0)
    inn = pd.to_numeric(df["Inn"], errors="coerce").fillna(0)
    estimated = (pa / 4.1) * 9
    is_dh = (inn == 0) | ((inn > 0) & (estimated / inn > 3))
    df.loc[is_dh, "Pos"] = "DH"
    return df


def filter_by_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    df = apply_dh_override(df)
    if position == "all" or "Pos" not in df.columns:
        return df
    if position == "OF":
        return df[df["Pos"].astype(str).str.upper().isin(["LF", "CF", "RF"])]
    return df[df["Pos"].astype(str).str.upper() == position.upper()]


def update_stat_default(i):
    stat = st.session_state[f"sc_stat_{i}"]
    st.session_state[f"sc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"sc_op_{i}"] = "<=" if stat in lower_better else ">="


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_year(year: int) -> pd.DataFrame:
    path = f"data/final/hitting_final_{year}.csv"
    try:
        df = pd.read_csv(path)
        df["Season"] = year
        bwar_df = load_bwar()
        if not bwar_df.empty:
            year_bwar = bwar_df[bwar_df["year_ID"] == year][["MLBAMID", "bWAR"]].copy()
            df = df.merge(year_bwar, on="MLBAMID", how="left")
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
#  Formatting
# ─────────────────────────────────────────────

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper = stat.upper()
    if upper in {"FRV", "OAA", "DRS"}:
        return f"{int(round(float(val)))}"
    if upper in {"fWAR", "OFF", "DEF", "BSR", "EV"}:
        v = float(val)
        return f"{v:.1f}" if abs(v - round(v)) >= 1e-9 else f"{int(round(v))}.0"
    if upper == "WPA":
        return f"{float(val):.2f}"
    if upper in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO"}:
        return f"{float(val):.3f}".lstrip("0") or ".000"
    if upper == "WRC+":
        return f"{int(round(float(val)))}"
    if "%" in stat or any(x in stat for x in ["Barrel", "Hard", "K%", "Swing", "Whiff"]):
        v = float(val)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"
    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def format_threshold(stat: str, val: float, op: str) -> str:
    lbl = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

for key, default in [
    ("sc_year",        current_year - 1),
    ("sc_min_pa",      300),
    ("sc_position",    "all"),
    ("sc_team",        "all"),
    ("sc_show_min_pa", True),
    ("sc_top10",       False),
    ("sc_val_0",       30.0),
    ("sc_val_1",       30.0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Controls
# ─────────────────────────────────────────────

col1, col2 = st.columns([0.5, 2])

with col1:
    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=1, horizontal=True, key="sc_num_stats")
    st.selectbox("Year", options=list(range(2025, 2014, -1)), key="sc_year")
    st.number_input("Min PA", min_value=0, max_value=20000, key="sc_min_pa")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        default_stat = "HR" if i == 0 else "SB" if i == 1 else STAT_ALLOWLIST[0]
        default_index = STAT_ALLOWLIST.index(default_stat) if default_stat in STAT_ALLOWLIST else 0

        new_stat = st.selectbox(
            f"Stat {i+1}", STAT_ALLOWLIST,
            key=f"sc_stat_{i}",
            index=default_index,
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed",
            on_change=update_stat_default,
            args=(i,),
        )

        op_col, val_col = st.columns([1, 2])
        with op_col:
            st.selectbox("Op", [">=", "<="], key=f"sc_op_{i}", index=0, label_visibility="collapsed")
        with val_col:
            RATE_STATS_3DP = {"AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "ISO", "BABIP"}
            if new_stat in RATE_STATS_3DP:
                step, fmt = 0.001, "%.3f"
            elif "%" in new_stat or new_stat == "EV" or "WAR" in new_stat:
                step, fmt = 0.1, "%.1f"
            else:
                step, fmt = 1.0, "%.0f"
            st.number_input(f"Value {i+1}", step=step, key=f"sc_val_{i}",
                            label_visibility="collapsed", format=fmt)

    st.selectbox("Position", options=list(POSITION_OPTIONS.keys()),
                 format_func=lambda x: POSITION_OPTIONS[x], key="sc_position")
    st.selectbox("Team", options=list(TEAM_OPTIONS.keys()),
                 format_func=lambda x: TEAM_OPTIONS[x], key="sc_team")
    st.checkbox("Show min PA",         key="sc_show_min_pa")
    st.checkbox("Only display top 10", key="sc_top10")

# ─────────────────────────────────────────────
#  Load & filter
# ─────────────────────────────────────────────

year_val     = int(st.session_state["sc_year"])
min_pa_val   = int(st.session_state["sc_min_pa"])
position_val = st.session_state["sc_position"]
team_val     = st.session_state["sc_team"]

df = load_final_year(year_val)

if df is None or df.empty:
    st.error(f"No data found for {year_val}.")
    st.stop()

# Min PA filter
if min_pa_val > 0 and "PA" in df.columns:
    df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa_val]

# Position filter (with DH override)
df = filter_by_position(df, position_val)

# Team filter
if team_val != "all" and "Team" in df.columns:
    target = normalize_team(team_val)
    df = df[df["Team"].astype(str).apply(normalize_team) == target]


# Team display
if "Team" in df.columns:
    df["TeamDisplay"] = df["Team"].astype(str).apply(get_team_display)
else:
    df["TeamDisplay"] = "N/A"

# Build active filters
active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"sc_stat_{i}")
    op   = st.session_state.get(f"sc_op_{i}", ">=")
    val  = float(st.session_state.get(f"sc_val_{i}", 0.0))
    if stat:
        active_filters.append((stat, op, val))

# Apply each filter
total_qualified = 0
if not df.empty:
    mask = pd.Series([True] * len(df), index=df.index)
    for stat, op, val in active_filters:
        if stat not in df.columns:
            continue
        col_vals = pd.to_numeric(df[stat], errors="coerce")
        compare_val = val
        # Handle pct stats stored as decimals vs whole numbers
        if stat in PCT_STATS:
            median_col = col_vals.median()
            if pd.notna(median_col) and median_col <= 1:
                if val > 1:
                    compare_val = val / 100
        mask = mask & (col_vals >= compare_val if op == ">=" else col_vals <= compare_val)

    df = df[mask]
    total_qualified = len(df)

    # Sort by first filter stat
    if active_filters:
        sort_stat, sort_op, _ = active_filters[0]
        if sort_stat in df.columns:
            asc = sort_stat in lower_better and sort_op == "<="
            df = df.sort_values(sort_stat, ascending=asc)

    display_limit = 10 if st.session_state.get("sc_top10") and total_qualified > 10 else MAX_DISPLAY
    if total_qualified > display_limit:
        df = df.head(display_limit)

# ─────────────────────────────────────────────
#  Build cards
# ─────────────────────────────────────────────

cards = []
for _, row in df.iterrows():
    name = str(row.get("Name", "")).strip()
    team = str(row.get("TeamDisplay", ""))

    stat_lines = []
    for stat, op, threshold in active_filters:
        val = row.get(stat, np.nan)
        if pd.notna(val):
            lbl = label_map.get(stat, stat)
            stat_lines.append(
                f'<span class="stat-label">{lbl}:</span> '
                f'<span class="stat-value">{html.escape(format_stat(stat, val))}</span>'
            )

    src = get_headshot(row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}" width="155" height="155" style="object-fit:cover;border-radius:6px;border:1px solid #e0e0e0;background:#f6f6f6;display:block;"/>'
    cards.append(f"""
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(name)}</div>
      <div class="player-team">{html.escape(team)}</div>
      {'<div class="player-stat-line">' + " | ".join(stat_lines) + "</div>" if stat_lines else ""}
    </div>""")

# ─────────────────────────────────────────────
#  Title & layout
# ─────────────────────────────────────────────

filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str  = ", ".join(filter_parts)
pos_suffix  = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
team_suffix = f"({team_val}) " if team_val != "all" else ""
title = f"{filter_str} in {year_val} {team_suffix}{pos_suffix}"

display_limit = 10 if st.session_state.get("sc_top10") and total_qualified > 10 else MAX_DISPLAY
overflow_note = (
    f'<div class="overflow-note">Showing top {display_limit} of {total_qualified} qualifying players</div>'
    if total_qualified > display_limit else ""
)
min_pa_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_pa_val} PA</div>'
    if st.session_state.get("sc_show_min_pa") else ""
)

body = "".join(cards) if cards else '<div style="padding:2rem;color:#999;text-align:center;">No players matched all filters. Try adjusting your thresholds.</div>'

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_pa_subtitle}
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
html, body {{ background: transparent; font-family: "Source Sans Pro", sans-serif; margin:0; padding:0; }}
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
.leaderboard-subtitle, .overflow-note {{
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    margin-top: -0.5rem;
}}
.players-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2rem 1rem;
}}
.player-card {{ flex: 0 0 155px; width: 155px; text-align: center; }}
.player-card img {{
    width: 155px; height: 155px;
    object-fit: cover; border-radius: 6px;
    border: 1px solid #e0e0e0; background: #f6f6f6;
}}
.player-name {{ font-weight: 800; font-size: 1rem; margin-top: 0.35rem; line-height: 1.2; }}
.player-team {{ color: #666; font-size: 0.8rem; margin-bottom: 0.25rem; }}
.player-stat-line {{ text-align: center; font-size: 0.95rem; margin-top: 0.15rem; }}
.stat-label {{ color: #888; font-size: 0.85rem; }}
.stat-value {{ font-weight: 800; font-size: 0.95rem; color: #1a1a1a; }}
.footer {{ display: flex; justify-content: space-between; margin-top: 1.5rem; padding: 0 4rem; }}
.footer p {{ margin: 0; font-size: 1rem; color: #888; flex: 1; text-align: center; }}
.footer p:first-child {{ text-align: left; }}
.footer p:last-child {{ text-align: right; }}
</style>
</head>
<body>{grid_html}</body>
</html>"""

with col2:
    components.html(full_html, height=est_height, scrolling=True)
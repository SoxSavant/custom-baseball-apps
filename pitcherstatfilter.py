import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import html
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Pitcher Stat Filter Leaderboard", layout="wide", page_icon="⚾")

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
    st.title("Pitcher Stat Filter Leaderboard")
with meta_col:
    st.markdown(
        '<div style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Constants — identical to pitcher leaderboard
# ─────────────────────────────────────────────

TEAM_ALIASES = {"ATH": "OAK", "ATH/OAK": "OAK", "OAK/ATH": "OAK"}
LOCAL_BWAR_FILE = Path(__file__).with_name("warpitchers.txt")
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

# Same stat list, label_map, lower_better as pitcher leaderboard
STAT_ALLOWLIST = [
    "fWAR", "bWAR",
    "ERA", "xERA", "FIP", "xFIP", "K%", "BB%", "K-BB%", "IP", "G", "GS",
    "Barrel%", "HardHit%", "EV", "GB%", "HR/9", "BABIP", "LOB%", "HR/FB",
    "SV", "AVG", "WHIP", "ERA-", "FIP-", "SIERA",
    "Chase%", "Whiff%", "WPA", "Clutch",
    "SO", "BB", "HBP", "HR", "QS", "CG", "ShO",
]

SUM_STATS = {
    "G", "GS", "HR", "BB", "SO", "HBP", "QS", "CG", "ShO", "SV", "WPA", "W", "L", "fWAR", "bWAR"
}
RATE_STATS = {
    "ERA", "xERA", "FIP", "xFIP", "K/9", "BB/9", "HR/9", "BABIP", "LOB%", "HR/FB",
    "K%", "BB%", "K-BB%", "AVG", "WHIP", "Barrel%", "HardHit%", "EV",
    "GB/FB", "GB%", "FB%", "SIERA", "Chase%", "Whiff%", "Pull%", "Cent%", "Oppo%", "Clutch",
    "ERA-", "FIP-",
}

label_map = {
    "EV": "Avg Exit Velo",
    "HardHit%": "Hard Hit%",
    "vFA (pi)": "vFA",
}

lower_better = {
    "ERA", "xERA", "FIP", "xFIP", "SIERA", "BB", "HBP", "HR",
    "BB/9", "HR/9", "BABIP", "HR/FB", "BB%", "AVG", "WHIP",
    "ERA-", "FIP-", "Barrel%", "HardHit%", "EV",
}

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


def update_stat_default(i):
    stat = st.session_state[f"pc_stat_{i}"]
    st.session_state[f"pc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"pc_op_{i}"] = "<=" if stat in lower_better else ">="


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_year(year: int) -> pd.DataFrame:
    path = f"data/final/pitching_final_{year}.csv"
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
#  Formatting — identical to pitcher leaderboard
# ─────────────────────────────────────────────

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()
    if upper_stat in {"fWAR", "EV", "bWAR"}:
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "K/9", "BB/9", "HR/9"}:
        return f"{float(val):.2f}"
    if upper_stat == "WHIP":
        return f"{float(val):.3f}"
    if upper_stat == "IP":
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    if upper_stat in {"ERA-", "FIP-"}:
        return f"{int(round(float(val)))}"
    if upper_stat in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"
    if upper_stat == "BABIP":
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


def format_threshold(stat: str, val: float, op: str) -> str:
    lbl = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

for key, default in [
    ("pc_year",        current_year - 1),
    ("pc_min_ip",      100),
    ("pc_team",        "all"),
    ("pc_show_ip",     False),
    ("pc_show_min_ip", True),
    ("pc_top10",       False),
    ("pc_val_0",       3.00),
    ("pc_val_1",       3.00),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Controls
# ─────────────────────────────────────────────

col1, col2 = st.columns([0.5, 2])

with col1:
    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=1, horizontal=True, key="pc_num_stats")
    st.selectbox("Year", options=list(range(2025, 2014, -1)), key="pc_year")
    st.number_input("Min IP", min_value=0, max_value=5000, key="pc_min_ip")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        default_stat = "ERA" if i == 0 else "FIP" if i == 1 else STAT_ALLOWLIST[0]
        default_index = STAT_ALLOWLIST.index(default_stat) if default_stat in STAT_ALLOWLIST else 0

        new_stat = st.selectbox(
            f"Stat {i+1}", STAT_ALLOWLIST,
            key=f"pc_stat_{i}",
            index=default_index,
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed",
            on_change=update_stat_default,
            args=(i,),
        )

        op_col, val_col = st.columns([1, 2])
        with op_col:
            st.selectbox("Op", ["<=", ">="], key=f"pc_op_{i}", index=0, label_visibility="collapsed")
        with val_col:
            RATE_3DP = {"WHIP", "BABIP"}
            RATE_2DP = {"ERA", "xERA", "FIP", "xFIP", "SIERA", "K/9", "BB/9", "HR/9", "HR/FB"}
            if new_stat in RATE_3DP:
                step, fmt = 0.001, "%.3f"
            elif new_stat in RATE_2DP:
                step, fmt = 0.01, "%.2f"
            elif "%" in new_stat or new_stat == "EV" or "WAR" in new_stat:
                step, fmt = 0.1, "%.1f"
            else:
                step, fmt = 1.0, "%.0f"
            st.number_input(f"Value {i+1}", step=step, key=f"pc_val_{i}",
                            label_visibility="collapsed", format=fmt)

    st.selectbox("Team", options=list(TEAM_OPTIONS.keys()),
                 format_func=lambda x: TEAM_OPTIONS[x], key="pc_team")
    st.checkbox("Show player IP",      key="pc_show_ip")
    st.checkbox("Show min IP",         key="pc_show_min_ip")
    st.checkbox("Only display top 10", key="pc_top10")

# ─────────────────────────────────────────────
#  Load & filter
# ─────────────────────────────────────────────

year_val   = int(st.session_state["pc_year"])
min_ip_val = int(st.session_state["pc_min_ip"])
team_val   = st.session_state["pc_team"]

df = load_final_year(year_val)

if df is None or df.empty:
    st.error(f"No data found for {year_val}.")
    st.stop()

# Min IP filter
if min_ip_val > 0 and "IP" in df.columns:
    df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip_val]

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
    stat = st.session_state.get(f"pc_stat_{i}")
    op   = st.session_state.get(f"pc_op_{i}", "<=")
    val  = float(st.session_state.get(f"pc_val_{i}", 0.0))
    if stat:
        active_filters.append((stat, op, val))

# Apply filters
total_qualified = 0
if not df.empty:
    mask = pd.Series([True] * len(df), index=df.index)
    for stat, op, val in active_filters:
        if stat not in df.columns:
            continue
        col_vals = pd.to_numeric(df[stat], errors="coerce")
        compare_val = val
        # Handle pct stats stored as decimals
        if stat in RATE_STATS:
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

    display_limit = 10 if st.session_state.get("pc_top10") and total_qualified > 10 else MAX_DISPLAY
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

    ip_val = row.get("IP", np.nan)
    ip_html = (
        f'<div class="player-ip">{format_stat("IP", ip_val)} IP</div>'
        if st.session_state.get("pc_show_ip") and pd.notna(ip_val) else ""
    )

    src = get_headshot(row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}" width="155" height="155" style="object-fit:cover;border-radius:6px;border:1px solid #e0e0e0;background:#f6f6f6;display:block;"/>'
    cards.append(f"""
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(name)}</div>
      <div class="player-team">{html.escape(team)}</div>
      {'<div class="player-stat-line">' + " | ".join(stat_lines) + "</div>" if stat_lines else ""}
      {ip_html}
    </div>""")

# ─────────────────────────────────────────────
#  Title
# ─────────────────────────────────────────────

filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str  = ", ".join(filter_parts)
team_suffix = f"({team_val}) " if team_val != "all" else ""
title = f"{filter_str} in {year_val} {team_suffix}".strip()

display_limit = 10 if st.session_state.get("pc_top10") and total_qualified > 10 else MAX_DISPLAY
overflow_note = (
    f'<div class="overflow-note">Showing top {display_limit} of {total_qualified} qualifying pitchers</div>'
    if total_qualified > display_limit else ""
)
min_ip_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_ip_val} IP</div>'
    if st.session_state.get("pc_show_min_ip") else ""
)

body = "".join(cards) if cards else '<div style="padding:2rem;color:#999;text-align:center;">No pitchers matched all filters. Try adjusting your thresholds.</div>'

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
.player-ip {{ color: #aaa; font-size: 0.8rem; margin-top: 0.1rem; }}
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
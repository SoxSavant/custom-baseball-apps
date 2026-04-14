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
#  Constants
# ─────────────────────────────────────────────

from p_utils import STAT_ALLOWLIST, SUM_STATS, RATE_STATS, TEAM_OPTIONS, format_stat
from p_utils import get_headshot, label_map, lower_better,  start_year
from p_utils import normalize_team, get_team_display, outs_to_ip, ip_to_outs

MAX_DISPLAY = 30

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

PCT_STATS = {
    "K%", "BB%", "K-BB%", "Chase%", "Whiff%", "Barrel%", "HardHit%",
    "GB%", "FB%", "LOB%", "HR/FB",
}

MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────



def update_stat_default(i):
    stat = st.session_state[f"pc_stat_{i}"]
    st.session_state[f"pc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"pc_op_{i}"] = "<=" if stat in lower_better else ">="


# ─────────────────────────────────────────────
#  Data loading & aggregation
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_year(year: int) -> pd.DataFrame:
    path = f"data/final/pitching_final_{year}.csv"
    try:
        df = pd.read_csv(path)
        df["Season"] = year
        return df
    except Exception:
        return pd.DataFrame()


def aggregate_player_group(grp: pd.DataFrame) -> dict:
    result: dict = {}

    if "Name" in grp.columns:
        result["Name"] = str(grp["Name"].dropna().iloc[0]) if not grp["Name"].dropna().empty else ""
    if "PlayerId" in grp.columns:
        ids = grp["PlayerId"].dropna()
        if not ids.empty:
            result["PlayerId"] = ids.iloc[0]
    if "MLBAMID" in grp.columns:
        ids = grp["MLBAMID"].dropna()
        if not ids.empty:
            result["MLBAMID"] = ids.iloc[0]

    if "Team" in grp.columns:
        teams = grp["Team"].dropna().astype(str).tolist()
        result["Team"] = (
            "2+ Teams" if any(get_team_display(t) == "2+ Teams" for t in teams)
            else "2+ Teams" if len({normalize_team(t) for t in teams if t.strip() and t.strip() != "- - -"}) > 1
            else (normalize_team(teams[0]) if teams else "N/A")
        )
    else:
        result["Team"] = "N/A"

    # IP: convert to outs, sum, convert back
    ip_outs_total = np.nan
    if "IP" in grp.columns:
        outs = pd.to_numeric(grp["IP"], errors="coerce").apply(ip_to_outs).dropna()
        if not outs.empty:
            ip_outs_total = outs.sum()
            result["IP"] = outs_to_ip(ip_outs_total)

    # Weight rate stats by IP outs
    if "IP" in grp.columns:
        weight = pd.to_numeric(grp["IP"], errors="coerce").apply(ip_to_outs).fillna(0)
    elif "TBF" in grp.columns:
        weight = pd.to_numeric(grp["TBF"], errors="coerce").fillna(0)
    else:
        weight = pd.Series(np.ones(len(grp)), index=grp.index)
    weight_total = weight.sum()

    numeric_cols = [
        col for col in grp.columns
        if pd.api.types.is_numeric_dtype(grp[col])
        and col not in {"PlayerId", "MLBAMID", "Season", "IP"}
    ]

    total_er = np.nan
    for col in numeric_cols:
        series = pd.to_numeric(grp[col], errors="coerce")
        if series.isna().all():
            continue
        if col == "ER":
            total_er = series.sum(skipna=True)
        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in RATE_STATS and weight_total > 0:
            result[col] = (series * weight).sum(skipna=True) / weight_total
        else:
            result[col] = series.mean(skipna=True)

    # Recompute ERA from totals
    if pd.notna(total_er) and not pd.isna(ip_outs_total) and ip_outs_total > 0:
        result["ERA"] = (total_er / (ip_outs_total / 3)) * 9

    return result


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(start_year: int, end_year: int, mode: str) -> pd.DataFrame:
    if mode == MODE_SINGLE:
        return load_final_year(start_year)

    frames = []
    for year in range(start_year, end_year + 1):
        df = load_final_year(year)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if mode == MODE_SPLIT:
        return combined

    # MODE_MULTI: aggregate by PlayerId
    if "PlayerId" not in combined.columns:
        return combined

    grouped_rows = []
    for _, grp in combined.groupby("PlayerId"):
        grouped_rows.append(aggregate_player_group(grp))

    return pd.DataFrame(grouped_rows)

def format_threshold(stat: str, val: float, op: str) -> str:
    lbl = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

from utils import get_dynamic_min_ip

for key, default in [
    ("pc_year",        current_year),
    ("pc_start_year",  current_year - 1),
    ("pc_end_year",    current_year),
    ("pc_mode",        MODE_SINGLE),
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

    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="pc_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year-1, -1)), key="pc_year")
        start_year = st.session_state["pc_year"]
        end_year   = st.session_state["pc_year"]

        if "pc_last_year" not in st.session_state:
            st.session_state.pc_last_year = start_year
        if start_year != st.session_state.pc_last_year:
            st.session_state["pc_min_ip"] = get_dynamic_min_ip(start_year)
            st.session_state.pc_last_year = start_year
    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year-1, -1)), key="pc_start_year")
        st.selectbox("End Year",   options=list(range(current_year, start_year-1, -1)), key="pc_end_year")
        start_year = st.session_state["pc_start_year"]
        end_year   = max(st.session_state["pc_end_year"], start_year)

    if "pc_min_ip" not in st.session_state:
        st.session_state["pc_min_ip"] = get_dynamic_min_ip(
            start_year if mode == MODE_SINGLE else current_year
        )

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
            RATE_2DP = {"ERA", "xERA", "FIP", "xFIP", "SIERA", "K/9", "BB/9", "HR/9", "HR/FB", "WPA", "Clutch"}
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

    team_disabled = (mode == MODE_MULTI)
    st.selectbox(
        "Team", options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x], key="pc_team",
        disabled=team_disabled,
        help="Team filter unavailable for multi-year span" if team_disabled else None,
    )

    st.checkbox("Show player IP",      key="pc_show_ip")
    st.checkbox("Show min IP",         key="pc_show_min_ip")
    st.checkbox("Only display top 10", key="pc_top10")

# ─────────────────────────────────────────────
#  Load & filter
# ─────────────────────────────────────────────

start_year = int(start_year)
end_year   = int(max(start_year, end_year))
min_ip_val = int(st.session_state["pc_min_ip"])
team_val   = "all" if team_disabled else st.session_state["pc_team"]

df = load_data(start_year, end_year, mode)

if df is None or df.empty:
    st.error(f"No data found for {start_year}–{end_year}.")
    st.stop()

# Min IP filter
if min_ip_val > 0 and "IP" in df.columns:
    df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip_val]

# Team filter (disabled in Multi mode)
if team_val != "all" and "Team" in df.columns:
    target = normalize_team(team_val)
    df = df[df["Team"].astype(str).apply(normalize_team) == target]

# Team display
if "Team" in df.columns:
    df["TeamDisplay"] = df["Team"].astype(str).apply(get_team_display)
else:
    df["TeamDisplay"] = "N/A"

# ─────────────────────────────────────────────
#  Apply stat filters
# ─────────────────────────────────────────────

active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"pc_stat_{i}")
    op   = st.session_state.get(f"pc_op_{i}", "<=")
    val  = float(st.session_state.get(f"pc_val_{i}", 0.0))
    if stat:
        active_filters.append((stat, op, val))

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
        mask = mask & (col_vals >= compare_val if op == ">=" else col_vals <= compare_val)

    df = df[mask]
    total_qualified = len(df)

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

    if mode == MODE_SPLIT and "Season" in row.index and pd.notna(row.get("Season")):
        team = f"{team} ({int(row['Season'])})"

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
#  Title & layout
# ─────────────────────────────────────────────

filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str   = ", ".join(filter_parts)
span_label   = f"{start_year}" if mode == MODE_SINGLE else f"{start_year}–{end_year}"
mode_label   = " (Single Season)" if mode == MODE_SPLIT else ""
team_suffix  = f" ({team_val})" if team_val != "all" else ""
title = f"{filter_str} in {span_label}{mode_label}{team_suffix}"

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
        <p>Data: FanGraphs, Bref</p>
    </div>
</div>
"""

card_count = len(cards)
est_rows   = max(1, (card_count + 4) // 5)
est_height = 180 + est_rows * 280 + 80

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
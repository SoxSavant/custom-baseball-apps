import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import html
from datetime import date

st.set_page_config(page_title="Hitter Stat Filter Leaderboard", layout="wide", page_icon="⚾")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
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

st.markdown("""
<style>
    @media only screen and (max-width: 600px) {
        [data-testid="stAppViewContainer"] h1 {
            font-size: 1.8rem !important;
        }

        .mobile-meta {
            font-size: 0.8rem !important;
            padding-top: 0.3rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Hitter Stat Filter Leaderboard")
with meta_col:
    st.markdown(
        '<div class = "mobile-meta" style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )
from h_utils import get_last_updated
current_year = date.today().year
last_updated = get_last_updated(current_year)
st.caption(f"2026 data last updated: {last_updated}")
# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
MAX_DISPLAY = 30

from h_utils import (STAT_ALLOWLIST, STAT_ROUND, RATE_STATS, format_stat, STAT_DEFAULTS, MAX_STATS,
get_headshot, label_map, lower_better,  start_year, POSITION_OPTIONS, TEAM_OPTIONS, normalize_team, 
get_team_display, filter_by_position, load_final_year,aggregate_player_group)



MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year

def update_stat_default(i):
    stat = st.session_state[f"sc_stat_{i}"]
    st.session_state[f"sc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"sc_op_{i}"] = "<=" if stat in lower_better else ">="


def load_data(start_year: int, end_year: int, mode: str, position: str = "all") -> pd.DataFrame:
    if mode == MODE_SINGLE:
        return load_final_year(start_year)

    frames = []
    for year in range(start_year, end_year + 1):
        df = load_final_year(year)
        if df is not None and not df.empty:
            if mode == MODE_SPLIT:
                df = filter_by_position(df,position_val)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if mode == MODE_SPLIT:
        return combined

    # MODE_MULTI: filter by position on raw rows first, then aggregate
    if "PlayerId" not in combined.columns:
        return combined

    combined = filter_by_position(combined, position)
    if combined.empty:
        return pd.DataFrame()

    return aggregate_player_group(combined)


# ─────────────────────────────────────────────
#  Formatting
# ─────────────────────────────────────────────


def format_threshold(stat: str, val: float, op: str) -> str:
    lbl = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


from utils import get_dynamic_min_pa

min_pa = get_dynamic_min_pa(current_year)

for key, default in [
    ("sc_year",        current_year),
    ("sc_start_year",  current_year - 1),
    ("sc_end_year",    current_year),
    ("sc_mode",        MODE_SINGLE),
    ("sc_min_type", "PA"),
    ("sc_position",    "all"),
    ("sc_team",        "all"),
    ("sc_show_min_pa", True),
    ("sc_show_player_pa", False),
    ("sc_top10",       False),
    ("sc_val_0",      160),
    ("sc_val_1",       .385),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([0.5, 2])

with col1:
    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=1, horizontal=True, key="sc_num_stats")

    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="sc_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year-1, -1)), key="sc_year")
        start_year = st.session_state["sc_year"]
        end_year   = st.session_state["sc_year"]

        if "sc_last_year" not in st.session_state:
            st.session_state.sc_last_year = start_year
        if start_year != st.session_state.sc_last_year:
            st.session_state["sc_min_pa"] = get_dynamic_min_pa(start_year)
            st.session_state.sc_last_year = start_year
    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year-1, -1)), key="sc_start_year")
        st.selectbox("End Year",   options=list(range(current_year, start_year-1, -1)), key="sc_end_year")
        start_year = st.session_state["sc_start_year"]
        end_year   = max(st.session_state["sc_end_year"], start_year)

    st.selectbox("Min Type", options=["PA", "Inn"], key="sc_min_type")

    if st.session_state["sc_min_type"] == "Inn":
        st.number_input("Min Inn", min_value=0, max_value=20000, value= 200, key="sc_min_inn")
    else:
        st.number_input("Min PA", min_value=0, max_value=20000, value=min_pa, key="sc_min_pa")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        default_stat = "wRC+" if i == 0 else "xwOBA" if i == 1 else STAT_ALLOWLIST[0]
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
            decimals = STAT_ROUND.get(new_stat, 0)
            step = 10 ** -decimals if decimals > 0 else 1.0
            fmt = f"%.{decimals}f"
            st.number_input(
                f"Value {i+1}", step=step, key=f"sc_val_{i}",
                label_visibility="collapsed", format=fmt,
            )

    st.selectbox("Position", options=list(POSITION_OPTIONS.keys()),
                 format_func=lambda x: POSITION_OPTIONS[x], key="sc_position")

    team_disabled = (mode == MODE_MULTI)
    st.selectbox(
        "Team", options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x], key="sc_team",
        disabled=team_disabled,
        help="Team filter unavailable for multi-year span" if team_disabled else None,
    )

    if st.session_state["sc_min_type"] == "PA":
        st.checkbox("Show Min PA", key="sc_show_min_pa")
    else:
        st.checkbox("Show Min Inn", key="sc_show_min_pa")
    if st.session_state["sc_min_type"] == "PA":
        st.checkbox("Show Player PA", key="sc_show_player_pa")
    else:
        st.checkbox("Show Player Inn", key="sc_show_player_pa")
    
    st.checkbox("Only display top 10", key="sc_top10")

use_inn = st.session_state.get("sc_min_type") == "Inn"
min_pa_val  = int(st.session_state.get("sc_min_pa", 0))
min_inn_val = int(st.session_state.get("sc_min_inn", 0))


position_val = st.session_state["sc_position"]
team_val     = "all" if team_disabled else st.session_state["sc_team"]

df = load_data(start_year, end_year, mode, position=position_val)

if df is None or df.empty:
    st.error(f"No data found for {start_year}–{end_year}.")
    st.stop()

if use_inn:
    if min_inn_val > 0 and "Inn" in df.columns:
        df = df[pd.to_numeric(df["Inn"], errors="coerce").fillna(0) >= min_inn_val]
else:
    if min_pa_val > 0 and "PA" in df.columns:
        df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa_val]

# Position filter — skip for MULTI (already applied pre-aggregation inside load_data)
if mode == MODE_SINGLE:
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


active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"sc_stat_{i}")
    op   = st.session_state.get(f"sc_op_{i}", ">=")
    val  = float(st.session_state.get(f"sc_val_{i}", 0.0))
    if stat:
        active_filters.append((stat, op, val))

total_qualified = 0
if not df.empty:
    mask = pd.Series([True] * len(df), index=df.index)
    for stat, op, val in active_filters:
        if stat not in df.columns:
            continue
        col_vals = pd.to_numeric(df[stat], errors="coerce")
        decimals = STAT_ROUND.get(stat, 0)
        col_vals = col_vals.round(decimals)
        compare_val = val
        if stat in RATE_STATS and "1350" not in stat:
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

    display_limit = 10 if st.session_state.get("sc_top10") and total_qualified > 10 else MAX_DISPLAY
    if total_qualified > display_limit:
        df = df.head(display_limit)

cards = []
for _, row in df.iterrows():
    name = str(row.get("Name", "")).strip()
    team = str(row.get("TeamDisplay", ""))

    # In Split mode, append the season year to the team label
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

    src = get_headshot(row)
    pa_val = row.get("PA", np.nan)
    inn_val = round(row.get("Inn", np.nan),1)
    if st.session_state.get("sc_show_player_pa"):
        if pd.notna(pa_val) and st.session_state.get("sc_min_type") == "PA":
            player_pa_html = (
            f'<div class="player-pa">{int(pa_val)} PA</div>'
        )
        elif pd.notna(inn_val) and st.session_state.get("sc_min_type") == "Inn":
            player_pa_html = (
            f'<div class="player-pa">{inn_val} Inn</div>'
        )
    else:
        player_pa_html = ""
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}" style="object-fit:cover;display:block;margin:0 auto;"/>'
    cards.append(f"""
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(name)}</div>
      <div class="player-team">{html.escape(team)}</div>
      {'<div class="player-stat-line">' + " | ".join(stat_lines) + "</div>" if stat_lines else ""}
      {player_pa_html}
    </div>""")

filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str   = ", ".join(filter_parts)
span_label   = f"{start_year}" if mode == MODE_SINGLE else f"{start_year}–{end_year}"
mode_label   = " (Single Season)" if mode == MODE_SPLIT else ""
pos_suffix   = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
team_suffix  = f" ({team_val})" if team_val != "all" else ""
middle_label = " in " if mode == MODE_SINGLE else ": "
title = f"{filter_str}{middle_label}{span_label}{mode_label}{team_suffix}{pos_suffix}"

display_limit = 10 if st.session_state.get("sc_top10") and total_qualified > 10 else MAX_DISPLAY
overflow_note = (
    f'<div class="overflow-note">Showing top {display_limit} of {total_qualified} qualifying players</div>'
    if total_qualified > display_limit else ""
)
if st.session_state.get("sc_show_min_pa"):
    if use_inn:
        min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_inn_val} Inn</div>'
    else:
        min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_pa_val} PA</div>'
else:
    min_pa_subtitle = ""

body = "".join(cards) if cards else '<div style="padding:2rem;color:#999;text-align:center;">No players matched all filters. Try adjusting your thresholds.</div>'

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_pa_subtitle}
    {overflow_note}
    <div class="players-grid">{body}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p>Data: FanGraphs • Baseball Reference • Baseball Savant</p>
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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
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
    box-sizing: border-box; /* Prevents padding from causing layout clipping */
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
.player-pa {{ color: #666; font-size: .9rem; }}
.stat-label {{ color: #888; font-size: 0.85rem; }}
.stat-value {{ font-weight: 800; font-size: 0.95rem; color: #1a1a1a; }}

.footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1.3rem -1rem 0 -1rem;
    padding: 0 3rem;
}}

.footer p {{
margin: 0;
font-size: 1rem;
color: #666;
white-space: nowrap;
}}

/* This block now safely fires and forces scaling on phone screens */
@media (max-width: 600px) {{
    .leaderboard-card {{
        width: 100% !important;
        padding: 1.5rem 0.5rem;
    }}
    .leaderboard-title {{
        font-size: 1.4rem;
        margin-bottom: 0.6rem;
    }}
    .leaderboard-subtitle, .overflow-note {{
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }}
    .players-grid {{
        gap: 1rem 0.35rem;
    }}
    .player-card {{
        flex: 0 0 calc(20% - 0.3rem);
        width: calc(20% - 0.3rem);
    }}
    .player-card img {{
        width: 100%;
        height: auto;
        aspect-ratio: 1 / 1;
    }}
    .player-name {{ 
        font-size: 0.65rem; 
        white-space: nowrap; 
        overflow: hidden; 
        text-overflow: ellipsis; 
    }}
    .player-team {{ font-size: 0.55rem; }}
    .player-stat-line {{ font-size: 0.65rem; }}
    .player-pa {{ font-size: 0.6rem; }}
    .stat-label {{ font-size: 0.6rem; }}
    .stat-value {{ font-size: 0.65rem; }}
    .footer p {{
        font-size: 0.65rem;
    }}

    .footer {{
    padding: 0 2rem;
    }}
}}
</style>
</head>
<body>{grid_html}</body>
</html>"""

with col2:
    components.html(full_html, height=est_height, scrolling=True)
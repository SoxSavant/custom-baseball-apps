import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import html
from datetime import date

from utils import TEAM_OPTIONS, LEAGUES

st.set_page_config(page_title="Pitcher Stat Filter Leaderboard", layout="wide", page_icon="⚾")

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
    st.title("Pitcher Stat Filter Leaderboard")
with meta_col:
    st.markdown(
        '<div class="mobile-meta" style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )

from p_utils import get_last_updated
current_year = date.today().year
last_updated = get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

from p_utils import (
    STAT_ALLOWLIST, format_stat,
    get_headshot, label_map, lower_better, start_year, STAT_DEFAULTS,
    normalize_team, get_team_display, load_final_year, aggregate_player_group,
    STAT_ROUND, STAT_DISPLAY_NAMES, PCT_STATS,
)

MAX_DISPLAY = 30

MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year

def update_stat_default(i):
    stat = st.session_state[f"pc_stat_{i}"]
    st.session_state[f"pc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"pc_op_{i}"] = "<=" if stat in lower_better else ">="


def load_data(s_year: int, e_year: int, mode: str) -> pd.DataFrame:
    if mode == MODE_SINGLE:
        return load_final_year(s_year)

    frames = []
    for year in range(s_year, e_year + 1):
        df = load_final_year(year)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if mode == MODE_SPLIT:
        return combined

    if "PlayerId" not in combined.columns:
        return combined

    return aggregate_player_group(combined)


def format_threshold(stat: str, val: float, op: str) -> str:
    lbl = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


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
    ("pc_view",        "Graphic"),
    ("pc_league", "All"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([0.5, 2])

with col1:
    view_mode = st.radio("View", ["Graphic", "Database"], key="pc_view", horizontal=True)

    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=1, horizontal=True, key="pc_num_stats")

    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="pc_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year - 1, -1)), key="pc_year")
        sel_start = st.session_state["pc_year"]
        sel_end   = st.session_state["pc_year"]

        if "pc_last_year" not in st.session_state:
            st.session_state.pc_last_year = sel_start
        if sel_start != st.session_state.pc_last_year:
            st.session_state["pc_min_ip"] = get_dynamic_min_ip(sel_start)
            st.session_state.pc_last_year = sel_start
    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key="pc_start_year")
        st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key="pc_end_year")
        sel_start = st.session_state["pc_start_year"]
        sel_end   = max(st.session_state["pc_end_year"], sel_start)

    if "pc_min_ip" not in st.session_state:
        st.session_state["pc_min_ip"] = get_dynamic_min_ip(
            sel_start if mode == MODE_SINGLE else current_year
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
            decimals = STAT_ROUND.get(new_stat, 0)
            step = 10 ** -decimals if decimals > 0 else 1.0
            fmt = f"%.{decimals}f"
            st.number_input(
                f"Value {i+1}", step=step, key=f"pc_val_{i}",
                label_visibility="collapsed", format=fmt,
            )

    team_disabled = (mode == MODE_MULTI)
    st.selectbox(
        "Team", options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x], key="pc_team",
        disabled=team_disabled,
        help="Team filter unavailable for multi-year span" if team_disabled else None,
    )

    league_disabled = (sel_start < 2013)
    st.selectbox(
        "League",
        options=LEAGUES.keys(),
        key="pc_league",
        disabled=team_disabled or league_disabled,
        help="League filter unavailable for years before 2013 due to possible inaccuracies" if league_disabled else None,
    )

    if view_mode == "Graphic":
        st.checkbox("Show player IP",      key="pc_show_ip")
        st.checkbox("Show min IP",         key="pc_show_min_ip")
        st.checkbox("Only display top 10", key="pc_top10")

sel_start  = int(sel_start)
sel_end    = int(max(sel_start, sel_end))
min_ip_val = int(st.session_state["pc_min_ip"])
team_val   = "all" if team_disabled else st.session_state["pc_team"]
league_val = "All" if team_disabled or league_disabled else st.session_state.get("pc_league", "All")

# ── Load & filter data (shared) ───────────────────────────────────────────────

df = load_data(sel_start, sel_end, mode)

if df is None or df.empty:
    st.error(f"No data found for {sel_start}–{sel_end}.")
    st.stop()

if min_ip_val > 0 and "IP" in df.columns:
    df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip_val]

if team_val != "all" and "Team" in df.columns:
    target = normalize_team(team_val)
    df = df[df["Team"].astype(str).apply(normalize_team) == target]

if league_val != "All" and "Team" in df.columns:
    league_teams = LEAGUES[league_val]
    df = df[
        df["Team"].astype(str).apply(
            lambda t: normalize_team(t) in league_teams
        )
    ]

if "Team" in df.columns:
    df["TeamDisplay"] = df["Team"].astype(str).apply(get_team_display)
else:
    df["TeamDisplay"] = "N/A"

# ── Build active filters & apply ─────────────────────────────────────────────

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
        decimals = STAT_ROUND.get(stat, 0)
        col_vals = col_vals.round(decimals)
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

# ── Shared label/span strings ─────────────────────────────────────────────────

filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str   = ", ".join(filter_parts)
span_label   = f"{sel_start}" if mode == MODE_SINGLE else f"{sel_start}–{sel_end}"
mode_label   = " (Single Season)" if mode == MODE_SPLIT else ""
team_suffix  = f" ({team_val})" if team_val != "all" else ""
league_suffix = f" ({league_val})" if league_val != "All" else ""
title        = f"{filter_str} in {span_label}{mode_label}{league_suffix}{team_suffix}"

# ── GRAPHIC VIEW ──────────────────────────────────────────────────────────────

with col2:
    if view_mode == "Graphic":
        display_limit = 10 if st.session_state.get("pc_top10") and total_qualified > 10 else MAX_DISPLAY
        df_graphic = df.head(display_limit)

        cards = []
        for _, row in df_graphic.iterrows():
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

            ip_val  = row.get("IP", np.nan)
            ip_html = (
                f'<div class="player-ip">{format_stat("IP", ip_val)} IP</div>'
                if st.session_state.get("pc_show_ip") and pd.notna(ip_val) else ""
            )

            src      = get_headshot(row)
            img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}" width="155" height="155" style="object-fit:cover;border-radius:6px;border:1px solid #e0e0e0;background:#f6f6f6;display:block;"/>'
            cards.append(f"""
            <div class="player-card">
              {img_html}
              <div class="player-name">{html.escape(name)}</div>
              <div class="player-team">{html.escape(team)}</div>
              {'<div class="player-stat-line">' + " | ".join(stat_lines) + "</div>" if stat_lines else ""}
              {ip_html}
            </div>""")

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
    background: #fff; border: 1px solid #d0d0d0; border-radius: 12px;
    padding: 3rem 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto; width: 100%; max-width: 900px; box-sizing: border-box;
}}
.leaderboard-title {{ font-weight: 900; font-size: 2.25rem; margin-bottom: 1.2rem; text-align: center; line-height: 1.2; }}
.leaderboard-subtitle, .overflow-note {{ text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 1rem; margin-top: -0.5rem; }}
.players-grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 2rem 1rem; }}
.player-card {{ flex: 0 0 155px; width: 155px; text-align: center; }}
.player-card img {{ width: 155px; height: 155px; object-fit: cover; border-radius: 6px; border: 1px solid #e0e0e0; background: #f6f6f6; }}
.player-name {{ font-weight: 800; font-size: 1rem; margin-top: 0.35rem; line-height: 1.2; }}
.player-team {{ color: #666; font-size: 0.8rem; margin-bottom: 0.25rem; }}
.player-stat-line {{ text-align: center; font-size: 0.95rem; margin-top: 0.15rem; }}
.player-ip {{ color: #666; font-size: .9rem; }}
.stat-label {{ color: #888; font-size: 0.85rem; }}
.stat-value {{ font-weight: 800; font-size: 0.95rem; color: #1a1a1a; }}
.footer {{ display: flex; justify-content: space-between; align-items: center; margin: 1.3rem -1rem 0 -1rem; padding: 0 3rem; }}
.footer p {{ margin: 0; font-size: 1rem; color: #666; white-space: nowrap; }}
@media (max-width: 600px) {{
    .leaderboard-card {{ width: 100% !important; padding: 1.5rem 0.5rem; }}
    .leaderboard-title {{ font-size: 1.4rem; margin-bottom: 0.6rem; }}
    .leaderboard-subtitle, .overflow-note {{ font-size: 0.9rem; margin-bottom: 0.8rem; }}
    .players-grid {{ gap: 1rem 0.35rem; }}
    .player-card {{ flex: 0 0 calc(20% - 0.3rem); width: calc(20% - 0.3rem); }}
    .player-card img {{ width: 100%; height: auto; aspect-ratio: 1 / 1; }}
    .player-name {{ font-size: 0.65rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .player-team {{ font-size: 0.55rem; }}
    .player-stat-line {{ font-size: 0.65rem; }}
    .player-ip {{ font-size: 0.6rem; }}
    .stat-label {{ font-size: 0.6rem; }}
    .stat-value {{ font-size: 0.65rem; }}
    .footer p {{ font-size: 0.65rem; }}
    .footer {{ padding: 0 2rem; }}
}}
</style>
</head>
<body>{grid_html}</body>
</html>"""

        components.html(full_html, height=est_height, scrolling=True)

    # ── DATABASE VIEW ─────────────────────────────────────────────────────────

    else:
        if df.empty:
            st.info("No pitchers matched all filters. Try adjusting your thresholds.")
            st.stop()

        filter_stats = [s for s, _, _ in active_filters]

        if mode == MODE_SPLIT:
            base_cols = [c for c in ["Name", "Team", "Season"] if c in df.columns]
        else:
            base_cols = [c for c in ["Name", "Team"] if c in df.columns]

        stat_cols = [s for s in filter_stats if s in df.columns and s not in base_cols]
        display   = df[base_cols + stat_cols].copy()
        display   = display.reset_index(drop=True)
        display.index += 1

        rename_map = {s: STAT_DISPLAY_NAMES.get(s, label_map.get(s, s)) for s in stat_cols}
        display    = display.rename(columns=rename_map)

        col_config: dict = {}
        for s in stat_cols:
            label = STAT_DISPLAY_NAMES.get(s, label_map.get(s, s))
            decimals = STAT_ROUND.get(s, 1)
            if s in PCT_STATS or "%" in s:
                col_config[label] = st.column_config.NumberColumn(label=label, format=f"%.{decimals}f%%")
            else:
                col_config[label] = st.column_config.NumberColumn(label=label, format=f"%.{decimals}f")

        st.caption(f"{total_qualified} pitchers — {title}")
        st.dataframe(display, width="stretch", height=700, column_config=col_config)

        st.markdown(
            "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem;'>"
            "Data: Baseball Reference · FanGraphs · Baseball Savant"
            "</div>",
            unsafe_allow_html=True,
        )
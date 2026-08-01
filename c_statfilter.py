import streamlit as st
import pandas as pd
import numpy as np
import html
from datetime import date
import h_utils
import p_utils




st.set_page_config(page_title="Stat Filter", layout="wide", page_icon="⚾")

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
    st.title("Stat Filter")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

type_mode = st.radio("Type", ["Hitting", "Pitching"], horizontal=True, key="cc_mode", label_visibility="collapsed")
is_hitting = (type_mode == "Hitting")
U = h_utils if is_hitting else p_utils
prefix = "hcc" if is_hitting else "pcc"

current_year = date.today().year
last_updated = U.get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

from utils import TEAM_OPTIONS, LEAGUES, get_dynamic_min_pa, get_dynamic_min_ip

MAX_DISPLAY = 30

from h_utils import ( POSITION_OPTIONS, normalize_team, filter_by_position)

STAT_ALLOWLIST = U.STAT_ALLOWLIST
TRUTHY_STRINGS = U.TRUTHY_STRINGS
get_headshot = U.get_headshot
label_map = U.label_map
lower_better = U.lower_better
start_year = U.start_year
format_stat = U.format_stat
load_final_year = U.load_final_year
aggregate_player_group = U.aggregate_player_group
STAT_ROUND = U.STAT_ROUND
STAT_DISPLAY_NAMES = U.STAT_DISPLAY_NAMES
RATE_STATS = U.RATE_STATS
STAT_DEFAULTS = U.STAT_DEFAULTS
get_team_display = U.get_team_display
PCT_STATS = U.PCT_STATS


MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year

def update_stat_default(i):
    stat = st.session_state[f"{prefix}_stat_{i}"]
    st.session_state[f"{prefix}_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"{prefix}_op_{i}"] = "<=" if stat in lower_better else ">="


def load_data(start_year: int, end_year: int, mode: str, position: str = "all") -> pd.DataFrame:
    if mode == MODE_SINGLE:
        return load_final_year(start_year)

    frames = []
    for year in range(start_year, end_year + 1):
        df = load_final_year(year)
        if df is not None and not df.empty:
            if mode == MODE_SPLIT:
                df = filter_by_position(df, position_val)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if mode == MODE_SPLIT:
        return combined

    if "PlayerId" not in combined.columns:
        return combined

    combined = filter_by_position(combined, position)
    if combined.empty:
        return pd.DataFrame()

    return aggregate_player_group(combined)


def format_threshold(stat: str, val: float, op: str) -> str:
    lbl = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"

min_pa = get_dynamic_min_pa(current_year)
min_ip = int(get_dynamic_min_ip(current_year))

default_val_0 = 150 if is_hitting else 3.00
default_val_1 = .375 if is_hitting else 3.00

default_stat_0 = "wRC+" if is_hitting else "ERA"
default_stat_1 = "xwOBA" if is_hitting else "xERA"


for key, default in [
    (f"{prefix}_year",        current_year),
    (f"{prefix}_start_year",  current_year - 1),
    (f"{prefix}_end_year",    current_year),
    (f"{prefix}_mode",        MODE_SINGLE),
    (f"{prefix}_min_type", "PA"),
    (f"{prefix}_position",    "all"),
    (f"{prefix}_team",        "all"),
    (f"{prefix}_show_min_pa", True),
    (f"{prefix}_show_min_ip", True),
    (f"{prefix}_show_player_pa", False),
    (f"{prefix}_show_player_ip", False),
    (f"{prefix}_top10",       False),
    (f"{prefix}_val_0",      default_val_0),
    (f"{prefix}_val_1",       default_val_1),
    (f"{prefix}_stat_0", default_stat_0),
    (f"{prefix}_stat_1", default_stat_1),
    (f"{prefix}_view",        "Graphic"),
    (f"{prefix}_league", "All"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([0.5, 2])

with col1:
    view_mode = st.radio("View", ["Graphic", "Database"], key=f"{prefix}_view", horizontal=True)

    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=1, horizontal=True, key=f"{prefix}_num_stats")

    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key=f"{prefix}_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year-1, -1)), key=f"{prefix}_year")
        start_year = st.session_state[f"{prefix}_year"]
        end_year   = st.session_state[f"{prefix}_year"]

        if f"{prefix}_last_year" not in st.session_state:
            st.session_state[f"{prefix}_last_year"] = start_year
        if start_year != st.session_state[f"{prefix}_last_year"]:
            st.session_state[f"{prefix}_min_pa"] = get_dynamic_min_pa(start_year)
            st.session_state[f"{prefix}_last_year"] = start_year
    else:
        start_options = list(range(current_year, start_year - 1, -1))
        current_start_val = st.session_state.get(f"{prefix}_start_year", current_year - 1)
        st.selectbox(
            "Start Year", options=start_options,
            index=start_options.index(current_start_val) if current_start_val in start_options else 0,
            key=f"{prefix}_start_year",
        )
        st.selectbox("End Year",   options=list(range(current_year, start_year-1, -1)), key=f"{prefix}_end_year")
        start_year = st.session_state[f"{prefix}_start_year"]
        end_year   = max(st.session_state[f"{prefix}_end_year"], start_year)

    if is_hitting:
        st.selectbox("Min Type", options=["PA", "Inn"], key=f"{prefix}_min_type")

    if is_hitting:
        if st.session_state[f"{prefix}_min_type"] == "Inn":
            st.number_input("Min Inn", min_value=0, max_value=20000, value=200, key=f"{prefix}_min_inn")
        else:
            st.number_input("Min PA", min_value=0, max_value=20000, value=min_pa, key=f"{prefix}_min_pa")
    else:
        st.number_input("Min IP (per season)", min_value=0, max_value=5000, value = min_ip, key=f"{prefix}_min_ip")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")

        new_stat = st.selectbox(
            f"Stat {i+1}", STAT_ALLOWLIST,
            key=f"{prefix}_stat_{i}",
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed",
            on_change=update_stat_default,
            args=(i,),
        )

        op_col, val_col = st.columns([1, 2])
        with op_col:

            st.selectbox("Op", [">=", "<="], key=f"{prefix}_op_{i}", index=0 if is_hitting else 1, label_visibility="collapsed")
        with val_col:
            decimals = STAT_ROUND.get(new_stat, 0)
            step = 10 ** -decimals if decimals > 0 else 1.0
            fmt = f"%.{decimals}f"
            st.number_input(
                f"Value {i+1}", step=step, key=f"{prefix}_val_{i}",
                label_visibility="collapsed", format=fmt,
            )

    if is_hitting:
        st.selectbox("Position", options=list(POSITION_OPTIONS.keys()),
                 format_func=lambda x: POSITION_OPTIONS[x], key=f"{prefix}_position")

    team_disabled = (mode == MODE_MULTI)
    st.selectbox(
        "Team", options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x], key=f"{prefix}_team",
        disabled=team_disabled,
        help="Team filter unavailable for multi-year span" if team_disabled else None,
    )

    league_disabled = (start_year < 2013)
    st.selectbox(
        "League",
        options=LEAGUES.keys(),
        key=f"{prefix}_league",
        disabled=team_disabled or league_disabled,
        help="League filter unavailable for years before 2013 due to possible inaccuracies" if league_disabled else None,
    )


    if view_mode == "Graphic" and is_hitting:
        if st.session_state[f"{prefix}_min_type"] == "PA":
            st.checkbox("Show Min PA", key=f"{prefix}_show_min_pa")
        else:
            st.checkbox("Show Min Inn", key=f"{prefix}_show_min_pa")
        if st.session_state[f"{prefix}_min_type"] == "PA":
            st.checkbox("Show Player PA", key=f"{prefix}_show_player_pa")
        else:
            st.checkbox("Show Player Inn", key=f"{prefix}_show_player_pa")
        st.checkbox("Only display top 10", key=f"{prefix}_top10")

    if view_mode == "Graphic" and not is_hitting:
        st.checkbox("Show min IP",         key=f"{prefix}_show_min_ip")
        st.checkbox("Show player IP",      key=f"{prefix}_show_player_ip")
        st.checkbox("Only display top 10", key=f"{prefix}_top10")

use_inn     = st.session_state.get(f"{prefix}_min_type") == "Inn"
min_pa_val  = int(st.session_state.get(f"{prefix}_min_pa", 0))
min_inn_val = int(st.session_state.get(f"{prefix}_min_inn", 0))
min_ip_val = int(st.session_state.get(f"{prefix}_min_ip", 0))

position_val = st.session_state[f"{prefix}_position"]
team_val     = "all" if team_disabled else st.session_state[f"{prefix}_team"]
league_val = "All" if team_disabled or league_disabled else st.session_state.get(f"{prefix}_league", "All")

df = load_data(start_year, end_year, mode, position=position_val)

if df is None or df.empty:
    st.error(f"No data found for {start_year}–{end_year}.")
    st.stop()

if is_hitting:
    if use_inn:
        if min_inn_val > 0 and "Inn" in df.columns:
            df = df[pd.to_numeric(df["Inn"], errors="coerce").fillna(0) >= min_inn_val]
    else:
        if min_pa_val > 0 and "PA" in df.columns:
            df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa_val]
else:
    if min_ip_val > 0 and "IP" in df.columns:
        df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip_val]


if mode == MODE_SINGLE and is_hitting:
    df = filter_by_position(df, position_val)

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
    stat = st.session_state.get(f"{prefix}_stat_{i}")
    op   = st.session_state.get(f"{prefix}_op_{i}", ">=")
    val  = float(st.session_state.get(f"{prefix}_val_{i}", 0.0))
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

# ── Shared label/span strings ─────────────────────────────────────────────────

filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str   = ", ".join(filter_parts)
span_label   = f"{start_year}" if mode == MODE_SINGLE else f"{start_year}–{end_year}"
mode_label   = " (Single Season)" if mode == MODE_SPLIT else ""
pos_suffix   = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
team_suffix  = f" ({team_val})" if team_val != "all" else ""
league_label = f" ({league_val})" if league_val != "All" else ""
middle_label = " – " if mode == MODE_SINGLE else ": "
title = f"{filter_str}{middle_label}{span_label}{mode_label}{league_label}{team_suffix}{pos_suffix}"


# ── GRAPHIC VIEW ──────────────────────────────────────────────────────────────

with col2:
    if view_mode == "Graphic":
        display_limit = 10 if st.session_state.get(f"{prefix}_top10") and total_qualified > 10 else MAX_DISPLAY
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

            src    = get_headshot(row)
            pa_val  = row.get("PA", np.nan)
            ip_val  = row.get("IP", np.nan)
            inn_val = round(row.get("Inn", np.nan), 1)
            if is_hitting:
                if st.session_state.get(f"{prefix}_show_player_pa"):
                    if pd.notna(pa_val) and st.session_state.get(f"{prefix}_min_type") == "PA":
                        player_pa_html = f'<div class="player-pa">{int(pa_val)} PA</div>'
                    elif pd.notna(inn_val) and st.session_state.get(f"{prefix}_min_type") == "Inn":
                        player_pa_html = f'<div class="player-pa">{inn_val} Inn</div>'
                    else:
                        player_pa_html = ""
                else:
                    player_pa_html = ""
            else:
                if st.session_state.get(f"{prefix}_show_player_ip"):
                    if pd.notna(ip_val):
                        player_pa_html = f'<div class="player-pa">{int(ip_val)} IP</div>'
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

        overflow_note = (
            f'<div class="overflow-note">Showing top {display_limit} of {total_qualified} qualifying players</div>'
            if total_qualified > display_limit else ""
        )
        if is_hitting:
            if st.session_state.get(f"{prefix}_show_min_pa"):
                if use_inn:
                    min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_inn_val} Inn</div>'
                else:
                    min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_pa_val} PA</div>'
            else:
                min_pa_subtitle = ""
        else:
            if st.session_state.get(f"{prefix}_show_min_ip"):
                min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_ip_val} IP</div>'

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
    box-sizing: border-box;
}}
.leaderboard-title {{ font-weight: 900; font-size: 2.25rem; margin-bottom: 1.2rem; text-align: center; line-height: 1.2; }}
.leaderboard-subtitle, .overflow-note {{ text-align: center; color: #888; font-size: 1.1rem; margin-bottom: 1rem; margin-top: -0.5rem; }}
.players-grid {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 2rem 1rem; }}
.player-card {{ flex: 0 0 155px; width: 155px; text-align: center; }}
.player-card img {{ width: 155px; height: 155px; object-fit: cover; border-radius: 6px; border: 1px solid #e0e0e0; background: #f6f6f6; }}
.player-name {{ font-weight: 800; font-size: 0.9rem; margin-top: 0.35rem; line-height: 1.2; }}
.player-team {{ color: #666; font-size: 0.8rem; margin-bottom: 0.25rem; }}
.player-stat-line {{ text-align: center; font-size: 0.95rem; margin-top: 0.15rem; }}
.player-pa {{ color: #666; font-size: .9rem; }}
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
    .player-pa {{ font-size: 0.6rem; }}
    .stat-label {{ font-size: 0.6rem; }}
    .stat-value {{ font-size: 0.65rem; }}
    .footer p {{ font-size: 0.65rem; }}
    .footer {{ padding: 0 2rem; }}
}}
</style>
</head>
<body>{grid_html}</body>
</html>"""

        st.iframe(full_html, height=est_height)

    # ── DATABASE VIEW ─────────────────────────────────────────────────────────

    else:
        if df.empty:
            st.info("No players matched all filters. Try adjusting your thresholds.")
            st.stop()

        filter_stats = [s for s, _, _ in active_filters]

        if mode == MODE_SPLIT:
            base_cols = [c for c in ["Name", "Team", "Season", "Pos"] if c in df.columns]
        else:
            base_cols = [c for c in ["Name", "Team", "Pos"] if c in df.columns]

        stat_cols = [s for s in filter_stats if s in df.columns and s not in base_cols]
        display   = df[base_cols + stat_cols].copy()
        display   = display.reset_index(drop=True)
        display.index += 1

        PCT_STAT_SET = PCT_STATS | {s for s in stat_cols if "%" in s}

        rename_map = {s: STAT_DISPLAY_NAMES.get(s, label_map.get(s, s)) for s in stat_cols}
        display    = display.rename(columns=rename_map)

        col_config: dict = {}
        for s in stat_cols:
            label = STAT_DISPLAY_NAMES.get(s, label_map.get(s, s))
            decimals = STAT_ROUND.get(s, 1)
            if s in PCT_STAT_SET:
                col_config[label] = st.column_config.NumberColumn(label=label, format=f"%.{decimals}f%%")
            else:
                col_config[label] = st.column_config.NumberColumn(label=label, format=f"%.{decimals}f")

        st.caption(f"{total_qualified} hitters – {title}")
        st.dataframe(display, width="stretch", height=700, column_config=col_config)

        st.markdown(
            "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem;'>"
            "Data: Baseball Reference · FanGraphs · Baseball Savant"
            "</div>",
            unsafe_allow_html=True,
        )
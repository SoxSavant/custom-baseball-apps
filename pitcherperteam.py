import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import html
from datetime import date

st.set_page_config(page_title="Pitcher Per Team Leaderboard", layout="wide", page_icon="⚾")

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
    st.title("Pitchers Per Team Leaderboard")
with meta_col:
    st.markdown(
        '<div class = "mobile-meta" style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )


from p_utils import (
    STAT_ALLOWLIST, RATE_STATS, format_stat, STAT_DEFAULTS, aggregate_player_group,
    label_map, lower_better, start_year, 
    normalize_team, get_team_display, load_final_year, TEAMS, STAT_ROUND
)
from utils import get_dynamic_min_ip, TEAM_MLB_IDS,  ALL_DIVISIONS, get_team_division, get_team_logo_url

MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year
MAX_TEAMS    = 9

def update_stat_default(i):
    stat = st.session_state[f"tc_stat_{i}"]
    st.session_state[f"tc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"tc_op_{i}"]  = "<=" if stat in lower_better else ">="


def format_threshold(stat: str, val: float, op: str) -> str:
    lbl       = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


def load_data(start_yr: int, end_yr: int, mode: str, position: str = "all") -> pd.DataFrame:
    if mode == MODE_SINGLE:
        return load_final_year(start_yr)

    frames = []
    for year in range(start_yr, end_yr + 1):
        df = load_final_year(year)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if mode == MODE_SPLIT:
        return combined

    # MODE_MULTI: aggregate each player per team across years
    if combined.empty or "PlayerId" not in combined.columns:
        return combined

    # Normalize team first, exclude multi-team season rows
    if "Team" in combined.columns:
        combined["TeamNorm"] = combined["Team"].astype(str).apply(normalize_team)
        combined = combined[combined["Team"].astype(str).str.strip() != "- - -"]

    grouped_rows = []
    group_cols = ["PlayerId", "TeamNorm"] if "TeamNorm" in combined.columns else ["PlayerId"]
    for _, grp in combined.groupby(group_cols):
        grouped_rows.append(aggregate_player_group(grp))

    return pd.DataFrame(grouped_rows)

for key, default in [
    ("tc_year",         current_year),
    ("tc_start_year",   current_year - 1),
    ("tc_end_year",     current_year),
    ("tc_mode",         MODE_SINGLE),
    ("tc_show_min_ip",  True),
    ("tc_show_player_ip", False),
    ("tc_show_all_teams", False),
    ("tc_leaders_only",False),
    ("tc_collapse_split", True),
    ("tc_val_0",        3.00),
    ("tc_val_1",        3.00),
]:
    if key not in st.session_state:
        st.session_state[key] = default

for div in ALL_DIVISIONS:
    key = f"tc_div_{div}"
    if key not in st.session_state:
        st.session_state[key] = True

col1, col2 = st.columns([0.5, 2])

with col1:
    num_stats = st.radio(
        "Number of stat filters", [1, 2, 3, 4],
        index=0, horizontal=True, key="tc_num_stats",
    )

    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="tc_mode")

    if mode == MODE_MULTI:
        st.caption("Multi-year excludes seasons ofplayers who switched teams mid-season in any year of the span.")
    
    if mode == MODE_SPLIT:
        st.checkbox("One card per team", key="tc_collapse_split")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year - 1, -1)), key="tc_year")
        s_year = st.session_state["tc_year"]
        e_year = st.session_state["tc_year"]

        if "tc_last_year" not in st.session_state:
            st.session_state.tc_last_year = s_year
        if s_year != st.session_state.tc_last_year:
            st.session_state["tc_min_ip"] = get_dynamic_min_ip(s_year)
            st.session_state.tc_last_year = s_year
    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key="tc_start_year")
        st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key="tc_end_year")
        s_year = st.session_state["tc_start_year"]
        e_year = max(st.session_state["tc_end_year"], s_year)

    if "tc_min_ip" not in st.session_state:
        st.session_state["tc_min_ip"] = get_dynamic_min_ip(
            s_year if mode == MODE_SINGLE else current_year
        )

    st.number_input("Min IP", min_value=0, max_value=20000, key="tc_min_ip")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        default_stat  = "ERA" if i == 0 else "FIP" if i == 1 else STAT_ALLOWLIST[0]
        default_index = STAT_ALLOWLIST.index(default_stat) if default_stat in STAT_ALLOWLIST else 0

        new_stat = st.selectbox(
            f"Stat {i+1}", STAT_ALLOWLIST,
            key=f"tc_stat_{i}",
            index=default_index,
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed",
            on_change=update_stat_default,
            args=(i,),
        )

        op_col, val_col = st.columns([1, 2])
        with op_col:
            st.selectbox("Op", ["<=", ">="], key=f"tc_op_{i}", index=0, label_visibility="collapsed")
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
            st.number_input(f"Value {i+1}", step=step, key=f"tc_val_{i}",
                            label_visibility="collapsed", format=fmt)

    st.checkbox("Show min IP",      key="tc_show_min_ip")
    st.checkbox("Show player IP",   key="tc_show_player_ip")
    st.checkbox("Show all 30 teams", key="tc_show_all_teams")
    st.checkbox("Only show leaders", key = "tc_leaders_only")

    st.markdown("**Divisions**")
    for div in ALL_DIVISIONS:
        st.checkbox(div, key=f"tc_div_{div}")

min_ip_val   = int(st.session_state["tc_min_ip"])

df = load_data(s_year, e_year, mode)

if df is None or df.empty:
    st.error(f"No data found for {s_year}–{e_year}.")
    st.stop()

# Min IP
if min_ip_val > 0 and "IP" in df.columns:
    df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip_val]

# Team display column
if "Team" in df.columns:
    df["TeamDisplay"] = df["Team"].astype(str).apply(get_team_display)
else:
    df["TeamDisplay"] = "N/A"

active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"tc_stat_{i}")
    op   = st.session_state.get(f"tc_op_{i}", ">=")
    val  = float(st.session_state.get(f"tc_val_{i}", 0.0))
    if stat:
        active_filters.append((stat, op, val))

# Apply stat filters
if not df.empty and active_filters:
    mask = pd.Series([True] * len(df), index=df.index)
    for stat, op, val in active_filters:
        if stat not in df.columns:
            continue
        col_vals    = pd.to_numeric(df[stat], errors="coerce")
        decimals = STAT_ROUND.get(stat, 0)
        col_vals = col_vals.round(decimals)
        compare_val = val
        if stat in RATE_STATS:
            median_col = col_vals.median()
            if pd.notna(median_col) and median_col <= 1 and val > 1:
                compare_val = val / 100
        mask = mask & (col_vals >= compare_val if op == ">=" else col_vals <= compare_val)
    df = df[mask]

# Exclude multi-team rows
if "Team" in df.columns:
    df = df[df["TeamDisplay"] != "2+ Teams"]

total_qualified = len(df)

show_all_teams   = st.session_state.get("tc_show_all_teams", False)
collapse_split   = st.session_state.get("tc_collapse_split", True)
active_divisions = [div for div in ALL_DIVISIONS if st.session_state.get(f"tc_div_{div}", True)]

sort_stat = active_filters[0][0] if active_filters else None
sort_op   = active_filters[0][1] if active_filters else ">="
sort_asc_tiebreak = sort_stat in lower_better and sort_op == "<=" if sort_stat else False

team_groups = []

if mode == MODE_SPLIT and not collapse_split:
    # One card per team+year combination
    group_keys = ["TeamDisplay", "Season"] if "Season" in df.columns else ["TeamDisplay"]
    qualifying_by_team_year: dict = {}
    if not df.empty and "TeamDisplay" in df.columns:
        for keys, grp in df.groupby(group_keys):
            team_abbrev = keys[0] if isinstance(keys, tuple) else keys
            year = int(keys[1]) if isinstance(keys, tuple) and len(keys) > 1 else None
            avg_val = np.nan
            if sort_stat and sort_stat in grp.columns:
                avg_val = pd.to_numeric(grp[sort_stat], errors="coerce").mean()
            player_df = grp.copy()
            if sort_stat and sort_stat in player_df.columns:
                asc = sort_stat in lower_better and sort_op == "<="
                player_df = player_df.sort_values(sort_stat, ascending=asc)
            key = (normalize_team(team_abbrev), year)
            qualifying_by_team_year[key] = {
                "avg_val": avg_val,
                "players": player_df,
                "year":    year,
            }

    if show_all_teams:
        years = list(range(s_year, e_year + 1))
        all_keys = [
            (a, y) for a in TEAM_MLB_IDS.keys() for y in years
            if get_team_division(a) in active_divisions
        ]
    else:
        all_keys = [k for k in qualifying_by_team_year.keys()
                    if get_team_division(k[0]) in active_divisions]

    for (abbrev, year) in all_keys:
        norm = normalize_team(abbrev)
        data = qualifying_by_team_year.get((norm, year), {"avg_val": np.nan, "players": pd.DataFrame(), "year": year})
        team_groups.append({
            "abbrev":       abbrev,
            "full_name":    TEAMS.get(norm, abbrev),
            "player_count": len(data["players"]),
            "avg_val":      data["avg_val"],
            "players":      data["players"],
            "year":         year,
        })

    team_groups.sort(
        key=lambda x: (
            -x["player_count"],
            x["avg_val"] if sort_asc_tiebreak else -x["avg_val"] if not np.isnan(x["avg_val"]) else 0,
            x["abbrev"],
            x["year"] or 0,
        )
    )
    if st.session_state.get("tc_leaders_only") and team_groups:
        top_count = team_groups[0]["player_count"]
        team_groups = [tg for tg in team_groups if tg["player_count"] == top_count]
    elif not show_all_teams:
        team_groups = team_groups[:MAX_TEAMS]

else:
    # Single season or multi-year span — one card per team
    qualifying_by_team: dict = {}
    if not df.empty and "TeamDisplay" in df.columns:
        for team_abbrev, grp in df.groupby("TeamDisplay"):
            avg_val = np.nan
            if sort_stat and sort_stat in grp.columns:
                avg_val = pd.to_numeric(grp[sort_stat], errors="coerce").mean()
            player_df = grp.copy()
            if sort_stat and sort_stat in player_df.columns:
                asc = sort_stat in lower_better and sort_op == "<="
                player_df = player_df.sort_values(sort_stat, ascending=asc)
            qualifying_by_team[normalize_team(team_abbrev)] = {
                "avg_val": avg_val,
                "players": player_df,
            }

    if show_all_teams:
        all_abbrevs = [a for a in TEAM_MLB_IDS.keys()]
    else:
        all_abbrevs = list(qualifying_by_team.keys())

    all_abbrevs = [a for a in all_abbrevs if get_team_division(a) in active_divisions]

    for abbrev in all_abbrevs:
        norm = normalize_team(abbrev)
        data = qualifying_by_team.get(norm, {"avg_val": np.nan, "players": pd.DataFrame()})
        team_groups.append({
            "abbrev":       abbrev,
            "full_name":    TEAMS.get(norm, abbrev),
            "player_count": len(data["players"]),
            "avg_val":      data["avg_val"],
            "players":      data["players"],
            "year":         None,
        })

    team_groups.sort(
        key=lambda x: (
            -x["player_count"],
            x["avg_val"] if sort_asc_tiebreak else -x["avg_val"] if not np.isnan(x["avg_val"]) else 0,
            x["abbrev"],
        )
    )
    if st.session_state.get("tc_leaders_only") and team_groups:
        top_count = team_groups[0]["player_count"]
        team_groups = [tg for tg in team_groups if tg["player_count"] == top_count]
    elif not show_all_teams:
        team_groups = team_groups[:MAX_TEAMS]


filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str   = ", ".join(filter_parts)
span_label   = f"{s_year}" if mode == MODE_SINGLE else f"{s_year}–{e_year}"
if mode == MODE_SPLIT and collapse_split:
    mode_label   = " (Combined Single Seasons)"
elif mode == MODE_SPLIT:
     mode_label   = " (Single Season)"
else:
    mode_label = " "
middle_label = " in " if mode == MODE_SINGLE else ": "
title_text   = f"Most Pitchers with {filter_str}{middle_label}{span_label}{mode_label}"

min_ip_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_ip_val} IP</div>'
    if st.session_state.get("tc_show_min_ip") else ""
)

show_player_ip = st.session_state.get("tc_show_player_ip", False)

# Build team card blocks
team_card_html = ""
if not team_groups:
    team_card_html = '<div style="padding:2rem;color:#999;text-align:center;">No teams matched all filters. Try adjusting your thresholds.</div>'
else:
    for rank, tg in enumerate(team_groups, start=1):
        abbrev      = tg["abbrev"]
        full_name   = tg["full_name"]
        count       = tg["player_count"]
        logo_url    = get_team_logo_url(abbrev)
        player_word = "player" if count == 1 else "players"

        year        = tg.get("year")
        year_label  = f" – {year}" if year is not None else ""

        logo_img = (
            f'<img src="{html.escape(logo_url)}" alt="{html.escape(abbrev)}" '
            f' style="object-fit:contain;display:block;"/>'
            if logo_url else
            f'<div style=display:flex;align-items:center;'
            f'justify-content:center;font-size:10px;font-weight:700;color:#888;">'
            f'{html.escape(abbrev)}</div>'
        )

        # Player rows — name on left, stats on right
        player_rows_html = ""
        for _, prow in tg["players"].iterrows():
            pname = str(prow.get("Name", "")).strip() 

            if collapse_split and mode == MODE_SPLIT and "Season" in prow.index and pd.notna(prow.get("Season")):
                pname = f'{html.escape(pname)} <span style="color:#aaa;font-weight:400;font-size:0.85rem;">– {int(prow["Season"])}</span>'
            else:
                pname = html.escape(pname)
            stat_parts = []
            for stat, op, threshold in active_filters:
                val = prow.get(stat, np.nan)
                if pd.notna(val):
                    lbl = label_map.get(stat, stat)
                    stat_parts.append(
                        f'<span class="p-stat-label">{html.escape(lbl)}</span>'
                        f'<span class="p-stat-val">{html.escape(format_stat(stat, val))}</span>'
                    )

            pa_html = ""
            if show_player_ip:
                ip_val = prow.get("IP", np.nan)
                if pd.notna(ip_val):
                    pa_html = f'<span class="p-ip">{int(ip_val)}</span>'

            stats_joined = "".join(stat_parts)
            player_rows_html += f"""
            <div class="player-row">
                <span class="p-name">{pname}</span>
                <span class="p-stats-right">{stats_joined}{pa_html}</span>
            </div>"""

        team_card_html += f"""
        <div class="team-card">
            <div class="team-header">
                <div class="team-logo-wrap">{logo_img}</div>
                <div class="team-meta">
                    <div class="team-abbrev-label">{html.escape(abbrev)}{html.escape(year_label)}</div>
                    <div class="team-badge">{count} {player_word}</div>
                </div>
                <div class="team-rank">#{rank}</div>
            </div>
            <div class="player-list">{player_rows_html}</div>
        </div>"""

num_teams_shown     = len(team_groups)
total_players_shown = sum(tg["player_count"] for tg in team_groups)

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title_text)}</div>
    {min_ip_subtitle}
    <div class="teams-grid">{team_card_html}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p>Data: FanGraphs • Baseball Reference • Baseball Savant</p>
    </div>
</div>
"""

# Height: title area + rows of cards (3 per row), each card height depends on player count
cards_per_row   = 3
num_rows        = max(1, -(-num_teams_shown // cards_per_row))  # ceiling division
avg_players     = total_players_shown / max(num_teams_shown, 1)
card_height_est = 90 + avg_players * 26
est_height      = 220 + int(num_rows * card_height_est) + 80

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style>
html, body {{
    background: transparent;
    font-family: "Source Sans Pro", sans-serif;
    margin: 0; padding: 0;
}}
.leaderboard-card {{
    background: #fff;
    border: 1px solid #d0d0d0;
    border-radius: 12px;
    padding: 2.5rem 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto;
    width: 100%;
    max-width: 900px;
    box-sizing: border-box;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2rem;
    margin-bottom: 0.4rem;
    text-align: center;
    line-height: 1.2;
    color: #111;
}}
.leaderboard-subtitle {{
    text-align: center;
    color: #6d7075;
    font-size: 1.1rem;
    margin-bottom: -0.2rem;
    margin-top: 0;
}}
.teams-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 1.25rem;
}}
.team-card {{
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    overflow: hidden;
    background: #fff;
    display: flex;
    flex-direction: column;
}}
.team-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 10px 8px;
    background: #fafafa;
    border-bottom: 1px solid #f0f0f0;
}}
.team-logo-wrap {{
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}}
.team-meta {{
    flex: 1;
    min-width: 0;
}}
.team-abbrev-label {{
    font-size: 1.5rem;
    font-weight: 800;
    color: #333;
    line-height: 1.1;
}}
.team-badge {{
    font-size: 1rem;
    font-weight: 700;
    color: #2e7d32;
    margin-top: 2px;
}}
.team-rank {{
    font-size: 0.9rem;
    font-weight: 700;
    color: #ccc;
    flex-shrink: 0;
    align-self: flex-start;
}}
.player-list {{
    padding: 4px 8px 6px;
    display: flex;
    flex-direction: column;
}}
.player-row {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #f4f4f4;
    gap: 4px;
}}
.player-row:last-child {{ border-bottom: none; }}
.p-name {{
    font-size: 1rem;
    font-weight: 700;
    color: #1a1a1a;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
    min-width: 0;
}}
.p-stats-right {{
    display: flex;
    align-items: baseline;
    gap: 4px;
    flex-shrink: 0;
    flex-wrap: wrap;
    justify-content: flex-end;
}}
.p-stat-label {{
    color: #aaa;
    font-size: .85rem;
    white-space: nowrap;
}}
.p-stat-val {{
    font-weight: 800;
    font-size: 1rem;
    color: #111;
    white-space: nowrap;
    margin-right: 5px;
}}
.p-pa {{
    color: #ccc;
    font-size: 0.9rem;
    margin-left: 2px;
}}
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

/* Compact Screenshot Rules for Small Screens */
@media (max-width: 600px) {{
    .leaderboard-card {{
        padding: 1rem 0.5rem;
    }}
    .leaderboard-title {{
        font-size: 1.2rem;
    }}
    .leaderboard-subtitle {{
        font-size: 0.8rem;
    }}
    .teams-grid {{
        grid-template-columns: repeat(3, 1fr); /* Locks the 3-column grid layout */
        gap: 4px; /* Tighter gutters */
    }}
    .team-header {{
        padding: 4px 4px 3px;
        gap: 4px;
    }}
     .team-logo-wrap {{
            width: 18px !important; 
            height: 18px !important;
        }}
    .team-logo-wrap img {{
            width: 100% !important;
            height: 100% !important;
            object-fit: contain !important;
    }}
    .team-abbrev-label {{
        font-size: 0.7rem; /* Scales text down dynamically */
    }}
    .team-badge {{
        font-size: 0.5rem;
    }}
    .team-rank {{
        font-size: 0.65rem;
    }}
    .player-list {{
        padding: 2px 4px;
    }}
    .p-name, .p-stat-val {{
        font-size: 0.5rem;
    }}
    .p-stat-label, .p-pa {{
        font-size: 0.5rem;
    }}
      .footer p {{
        font-size: 0.7rem;
    }}
    .footer {{
    margin-top: 1rem;
    }}
    .leaderboard-subtitle {{
    margin-bottom: -0.7rem;
}}
}}
</style>
</head>
<body>{grid_html}</body>
</html>"""

with col2:
    components.html(full_html, height=est_height, scrolling=True)
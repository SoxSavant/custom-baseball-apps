import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import html
import re
from datetime import date

st.set_page_config(page_title="Custom Pitching Leaderboard", layout="wide", page_icon="⚾")

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
    st.title("Custom Pitcher Leaderboard")
with meta_col:
    st.markdown(
        """
        <div class = "mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
from h_utils import get_last_updated
current_year = date.today().year
last_updated = get_last_updated(current_year)
st.caption(f"2026 data last updated: {last_updated}")
from p_utils import (STAT_ALLOWLIST, start_year,
get_headshot, label_map, lower_better,  TEAM_OPTIONS, normalize_team, get_team_display, 
format_stat,load_final_year, aggregate_player_group)


MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year


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

    return aggregate_player_group(combined)
    

from utils import get_dynamic_min_ip

min_ip = get_dynamic_min_ip(current_year)

for key, default in [
    ("pl_year",           current_year),
    ("pl_start_year",     current_year - 1),
    ("pl_end_year",       current_year),
    ("pl_stat",           "fWAR"),
    ("pl_min_ip",         min_ip),
    ("pl_team",           "all"),
    ("pl_mode",           MODE_SINGLE),
    ("pl_sort_worst",     False),
    ("pl_show_min_ip",    True),
    ("pl_show_player_ip", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

stat = st.selectbox(
    "Stat", STAT_ALLOWLIST, key="pl_stat",
    format_func=lambda x: label_map.get(x, x),
)

col1, col2 = st.columns([.5, 2])

with col1:
    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="pl_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year-1, -1)), key="pl_year")
        start_year = st.session_state["pl_year"]
        end_year   = st.session_state["pl_year"]

        if "last_year" not in st.session_state:
            st.session_state.last_year = start_year

        if start_year != st.session_state.last_year:
            st.session_state["pl_min_ip"] = get_dynamic_min_ip(start_year)
            st.session_state.last_year = start_year

    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year-1, -1)), key="pl_start_year", 
                      )
        st.selectbox("End Year", options=list(range(current_year, start_year-1, -1)), key="pl_end_year", 
                     )
    
        start_year = st.session_state["pl_start_year"]
        end_year   = max(st.session_state["pl_end_year"], start_year)

    st.number_input("Min IP", min_value=0, max_value=5000, key="pl_min_ip")

    team_disabled = (mode == MODE_MULTI)
    st.selectbox(
        "Team", options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x], key="pl_team",
        disabled=team_disabled,
        help="Team filter unavailable for Multi-Year Span mode." if team_disabled else None,
    )

    st.checkbox("Show worst",     key="pl_sort_worst")
    st.checkbox("Show min IP",    key="pl_show_min_ip")
    st.checkbox("Show player IP", key="pl_show_player_ip")

start_year = int(start_year)
end_year   = int(max(start_year, end_year))
min_ip_val = int(st.session_state.get("pl_min_ip", 0))
team_val   = "all" if team_disabled else st.session_state.get("pl_team", "all")


df = load_data(start_year, end_year, mode)

if df is None or df.empty:
    st.error(f"No data found for {start_year}–{end_year}.")
    st.stop()

# Min IP filter
if min_ip_val > 0 and "IP" in df.columns:
    df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip_val]

# Team filter
if team_val != "all" and "Team" in df.columns:
    target = normalize_team(team_val)
    df = df[df["Team"].astype(str).apply(lambda t: normalize_team(t) == target)]

# Team display column
if "Team" in df.columns:
    df["TeamDisplay"] = df["Team"].astype(str).apply(get_team_display)
else:
    df["TeamDisplay"] = "N/A"

# Sort & top 10
if stat not in df.columns:
    st.error(f"Stat '{stat}' not found in dataset for {start_year}–{end_year}.")
    st.stop()

df[stat] = pd.to_numeric(df[stat], errors="coerce")
is_lower_better = stat in lower_better
show_worst = st.session_state.get("pl_sort_worst", False)
ascending = (is_lower_better and not show_worst) or (not is_lower_better and show_worst)
df = df.sort_values(by=stat, ascending=ascending).dropna(subset=[stat]).head(10)


cards = []
for _, row in df.iterrows():
    name        = str(row.get("Name", "")).strip()
    team        = str(row.get("TeamDisplay", ""))
    raw_val     = row.get(stat, np.nan)
    display_val = format_stat(stat, raw_val)

    if mode == MODE_SPLIT and "Season" in row.index and pd.notna(row.get("Season")):
        team = f"{team} ({int(row['Season'])})"

    ip_val = row.get("IP", np.nan)
    ip_html = (
        f'<div class="player-ip">{format_stat("IP", ip_val)} IP</div>'
        if st.session_state.get("pl_show_player_ip") and pd.notna(ip_val) else ""
    )

    src = get_headshot(row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}"/>'
    cards.append(f"""
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(name)}</div>
      <div class="player-team">{html.escape(team)}</div>
      <div class="player-stat">{html.escape(display_val)}</div>
      {ip_html}
    </div>
    """)


span_label  = f"{start_year}" if mode == MODE_SINGLE else f"{start_year}–{end_year}"
title_label = label_map.get(stat, stat)
team_label  = TEAM_OPTIONS.get(team_val, "") if team_val != "all" else ""
mode_label  = " Single Season" if mode == MODE_SPLIT else ""
worst_label = " Worst " if show_worst else ""
title = re.sub(r"  +", " ", f"{span_label}{mode_label} {team_label}{worst_label} {title_label} Leaders".strip())

min_pa_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_ip_val} IP</div>'
    if st.session_state.get("pl_show_min_ip") else ""
)


grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_pa_subtitle}
    <div class="players-grid">{"".join(cards)}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p>Data: FanGraphs • Baseball Reference • Baseball Savant</p>
    </div>
</div>
"""

full_html = f"""
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800&display=swap" rel="stylesheet">
<meta charset="utf-8" />
<style>
html, body {{ background: transparent; font-family: "Source Sans Pro", sans-serif; margin:0; padding:0; }}
.leaderboard-card {{
    background: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 12px;
    padding: 3rem 4rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto;
    width: 100%;
    max-width: 900px;
    box-sizing: border-box;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.4rem;
    margin-bottom: 2rem;
    text-align: center;
}}
.leaderboard-subtitle{{
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    margin-top: -1.5rem;
}}
.players-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    justify-content: start;
    justify-items: center;
    row-gap: 1rem;
    column-gap: 4rem;
}}
.player-card {{ text-align: center; }}
.player-card img {{
    width: 155px;
    height: 155px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    background: #f6f6f6;
}}
.player-name {{ font-weight: 800; margin-top: 0.35rem; font-size: 1.18rem; }}
.player-team {{ color: #666; font-size: 0.85rem; }}
.player-stat {{ font-weight: 900; font-size: 1.5rem; margin-top: 0.25rem; }}
.player-ip {{ color: #666; font-size: .85rem; }}
html, body {{ margin: 0; padding: 0; background: transparent; width: 100%; }}

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

@media (max-width: 600px) {{

    .leaderboard-card {{
        padding: 1rem 0.75rem;
        border-radius: 10px;
    }}

    .leaderboard-title {{
        font-size: 1.35rem;
        margin-bottom: 0.6rem;
    }}

    .leaderboard-subtitle {{
        font-size: 0.85rem;
        margin-top: -0.4rem;
    }}

    .players-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    grid-auto-rows: auto;
    gap: 0.5rem;
}}

    .player-card {{
        min-width: 130px;
        flex: 0 0 auto;
        scroll-snap-align: start;
    }}

    .player-card img {{
    width: 80px;
    height: 80px;
}}

    .player-name {{ font-size: 0.7rem; }}
    .player-team {{ font-size: 0.7rem; }}
    .player-stat {{ font-size: .9rem; }}
    .player-pa {{ font-size: 0.75rem; }}

      .footer p {{
        font-size: 0.7rem;
    }}
    .footer {{
    margin-top: 1rem;
    }}
}}
</style>
</head>
<body>{grid_html}</body>
</html>
"""

with col2:
    components.html(full_html, height=800)
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import unicodedata
import html
import re
from datetime import date

st.set_page_config(page_title="Custom Hitting Leaderboard", layout="wide", page_icon="⚾")

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
    st.title("Custom Hitter Leaderboard")
with meta_col:
    st.markdown(
        """
        <div class = "mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


from h_utils import (STAT_ALLOWLIST, format_stat, start_year,
get_headshot, label_map, lower_better, load_final_year,
POSITION_OPTIONS, TEAM_OPTIONS, normalize_team, get_team_display, filter_by_position, aggregate_player_group)

MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year


def normalize_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\xa0", " ").strip()
    try:
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(cleaned.split()).lower()


# ─────────────────────────────────────────────
#  Main data builder
# ─────────────────────────────────────────────


def load_data(start_year: int, end_year: int, mode: str, position: str = "all") -> pd.DataFrame:
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
    # mode is #multi 
    combined = filter_by_position(combined, position)
    if combined.empty:
        return pd.DataFrame

    grouped_rows = []
    for _, grp in combined.groupby("PlayerId"):
        grouped_rows.append(aggregate_player_group(grp))

    return pd.DataFrame(grouped_rows)
    

from utils import get_dynamic_min_pa

min_pa = get_dynamic_min_pa(current_year)


for key, default in [
    ("hl_year", current_year),
    ("hl_start_year", current_year - 1),
    ("hl_end_year", current_year),
    ("hl_stat", "fWAR"),
    ("hl_min_type", "PA"),
    ("hl_position", "all"),
    ("hl_team", "all"),
    ("hl_mode", MODE_SINGLE),
    ("hl_show_player_pa", False),
    ("hl_sort_worst", False),
    ("hl_show_min_pa", True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

stat = st.selectbox(
    "Stat", STAT_ALLOWLIST, key="hl_stat",
    format_func=lambda x: label_map.get(x, x),
)

col1, col2 = st.columns([.5, 2])

with col1:
    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="hl_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year-1, -1)), key="hl_year")
        start_year = st.session_state["hl_year"]
        end_year   = st.session_state["hl_year"]
        
        if "last_year" not in st.session_state:
            st.session_state.last_year = start_year

        if start_year != st.session_state.last_year:
            st.session_state["hl_min_pa"] = get_dynamic_min_pa(start_year)
            st.session_state.last_year = start_year

    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year-1, -1)), key="hl_start_year")
        st.selectbox("End Year", options=list(range(current_year, start_year-1, -1)), key="hl_end_year")
        start_year = st.session_state["hl_start_year"]
        end_year   = max(st.session_state["hl_end_year"], start_year)
    
    st.selectbox("Min Type", options=["PA", "Inn"], key="hl_min_type")

    if st.session_state["hl_min_type"] == "Inn":
        st.number_input("Min Inn", min_value=0, max_value=20000, value= 200, key="hl_min_inn")
    else:
        st.number_input("Min PA", min_value=0, max_value=20000, value=min_pa, key="hl_min_pa")

    st.selectbox("Position", options=list(POSITION_OPTIONS.keys()),
                 format_func=lambda x: POSITION_OPTIONS[x], key="hl_position")

    team_disabled = (mode == MODE_MULTI)
    st.selectbox("Team", options=list(TEAM_OPTIONS.keys()),
                 format_func=lambda x: TEAM_OPTIONS[x], key="hl_team",
                 disabled=team_disabled,
                 help="Team filter unavailable for multi-year span" if team_disabled else None)

    st.checkbox("Show worst",     key="hl_sort_worst")
    if st.session_state["hl_min_type"] == "PA":
        st.checkbox("Show Min PA", key="hl_show_min_pa")
    else:
        st.checkbox("Show Min Inn", key="hl_show_min_pa")
    if st.session_state["hl_min_type"] == "PA":
        st.checkbox("Show Player PA", key="hl_show_player_pa")
    else:
        st.checkbox("Show Player Inn", key="hl_show_player_pa")

position_val = st.session_state.get("hl_position", "all")
team_val     = "all" if team_disabled else st.session_state.get("hl_team", "all")


# load data

df = load_data(start_year, end_year, mode, position_val)
if df is None or df.empty:
    st.error(f"No data found for {start_year}–{end_year}.")
    st.stop()

use_inn = st.session_state.get("hl_min_type") == "Inn"
min_pa_val  = int(st.session_state.get("hl_min_pa", 0))
min_inn_val = int(st.session_state.get("hl_min_inn", 0))

if use_inn:
    if min_inn_val > 0 and "Inn" in df.columns:
        df = df[pd.to_numeric(df["Inn"], errors="coerce").fillna(0) >= min_inn_val]
else:
    if min_pa_val > 0 and "PA" in df.columns:
        df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa_val]

# After load_data call:
if mode != MODE_MULTI:
    df = filter_by_position(df, position_val)

# Team filter (single/split season only)
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
stat_lower_better = stat in lower_better
sort_worst = st.session_state.get("hl_sort_worst", False)
ascending = (stat_lower_better and not sort_worst) or (not stat_lower_better and sort_worst)
df = df.sort_values(by=stat, ascending=ascending).dropna(subset=[stat]).head(10)

# ─────────────────────────────────────────────
#  Build cards
# ─────────────────────────────────────────────

cards = []
for _, row in df.iterrows():
    name = str(row.get("Name", "")).strip()
    team = str(row.get("TeamDisplay", ""))
    raw_val = row.get(stat, np.nan)
    display_val = format_stat(stat, raw_val)

    if mode == MODE_SPLIT and "Season" in row.index and pd.notna(row.get("Season")):
        team = f"{team} ({int(row['Season'])})"

    src = get_headshot(row)
    pa_val = row.get("PA", np.nan)
    inn_val = round(row.get("Inn", np.nan), 1)
    if st.session_state.get("hl_show_player_pa"):
        if pd.notna(pa_val) and st.session_state.get("hl_min_type") == "PA":
            player_pa_html = (
            f'<div class="player-pa">{int(pa_val)} PA</div>'
        )
        elif pd.notna(inn_val) and st.session_state.get("hl_min_type") == "Inn":
            player_pa_html = (
            f'<div class="player-pa">{inn_val} Inn</div>'
        )
    else:
        player_pa_html = ""
    


    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}"/>'
    cards.append(f'''
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(name)}</div>
      <div class="player-team">{html.escape(team)}</div>
      <div class="player-stat">{html.escape(display_val)}</div>
      {player_pa_html}
    </div>
    ''')

span_label = f"{start_year}" if mode == MODE_SINGLE else f"{start_year}–{end_year}"
title_label = label_map.get(stat, stat)
pos_suffix = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
team_label = TEAM_OPTIONS.get(team_val, "") if team_val != "all" else ""
mode_label = " Single Season" if mode == MODE_SPLIT else ""
worst_label = "Worst " if sort_worst else ""

title = re.sub(r"  +", " ", f"{span_label}{mode_label} {team_label} {worst_label}{title_label} Leaders{pos_suffix}".strip())


if st.session_state.get("hl_show_min_pa"):
    if use_inn:
        min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_inn_val} Inn</div>'
    else:
        min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_pa_val} PA</div>'
else:
    min_pa_subtitle = ""


grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_pa_subtitle}
    <div class="players-grid">{''.join(cards)}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p></p>
        <p>Data: FanGraphs, Bref</p>
    </div>
</div>
"""

full_html = f"""
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800&display=swap" rel="stylesheet">
<meta charset="utf-8" />
<style>
html, body {{
    background: transparent;
    font-family: "Source Sans Pro", sans-serif;
    margin: 0;
    padding: 0;
}}

/* ───────── DESKTOP ───────── */

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

.leaderboard-subtitle {{
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    margin-top: -1.5rem;
}}

.players-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
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
.player-pa {{ color: #666; font-size: 1rem; }}

.footer {{
    display: flex;
    justify-content: space-between;
    margin-top: .5rem;
}}

.footer p {{
    margin: 0;
    font-size: 0.9rem;
    color: #666;
    flex: 1;
    text-align: center;
}}

/* ───────── MOBILE ───────── */

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
}}
</style>
</head>
<body>{grid_html}</body>
</html>
"""

with col2:
    components.html(full_html, height=800)
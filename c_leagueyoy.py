import streamlit as st
import pandas as pd
import numpy as np
import html
from datetime import date
import h_utils
import p_utils
import unicodedata

from utils import TEAM_OPTIONS


st.set_page_config(page_title="League Year-over-Year", layout="wide", page_icon="⚾")

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
    st.title("League Year-over-Year")
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

from h_utils import ( POSITION_OPTIONS, normalize_team,  filter_by_position,)

STAT_ALLOWLIST = U.STAT_ALLOWLIST
TRUTHY_STRINGS = U.TRUTHY_STRINGS
get_headshot = U.get_headshot
label_map = U.label_map
lower_better = U.lower_better
start_year = U.start_year
format_stat_yoy = U.format_stat_yoy
load_final_year = U.load_final_year
STAT_ROUND = U.STAT_ROUND
get_team_display = U.get_team_display
PCT_STATS = U.PCT_STATS
load_risers_data = U.load_risers_data

def normalize_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\xa0", " ").strip()
    try:
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(cleaned.split()).lower()

from utils import get_dynamic_min_pa, get_dynamic_min_ip

min_pa_end   = get_dynamic_min_pa(current_year)
min_pa_start = 502

min_ip_end = int(get_dynamic_min_ip(current_year))
min_ip_start = 162

default_stat = "wRC+" if is_hitting else "ERA"

for key, default in [
    (f"{prefix}_start_year",     current_year - 1),
    (f"{prefix}_end_year",       current_year),
    (f"{prefix}_stat",           default_stat),
    (f"{prefix}_min_type",       "PA"),
    (f"{prefix}_position",       "all"),
    (f"{prefix}_team",           "all"),
    (f"{prefix}_show_fallers",   False),
    (f"{prefix}_show_min_pa",    True),
    (f"{prefix}_show_player_pa", False),
    (f"{prefix}_show_min_ip",    True),
    (f"{prefix}_show_player_ip", False),
    (f"{prefix}_view", "Graphic"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.get(f"{prefix}_stat") not in STAT_ALLOWLIST:
    st.session_state[f"{prefix}_stat"] = "wRC+"


col1, col2 = st.columns([0.5, 2])

with col1:
    view_mode = st.radio("View", ["Graphic", "Database"], key=f"{prefix}_view", horizontal=True)
    stat = st.selectbox(
    "Stat", STAT_ALLOWLIST, key=f"{prefix}_stat",
    format_func=lambda x: label_map.get(x, x),
)
    st.selectbox("Start Year", options=list(range(current_year - 1, start_year - 1, -1)), key=f"{prefix}_start_year")
    st.selectbox("End Year",   options=list(range(current_year,     start_year - 1, -1)), key=f"{prefix}_end_year")
    start_year = st.session_state[f"{prefix}_start_year"]
    end_year   = st.session_state[f"{prefix}_end_year"]
    if end_year <= start_year:
        st.warning("End Year must be greater than Start Year.")

    if is_hitting:
        st.selectbox("Min Type", options=["PA", "Inn"], key=f"{prefix}_min_type")
        use_inn = st.session_state[f"{prefix}_min_type"] == "Inn"

    if is_hitting:
        if use_inn:
            st.number_input("Start Year Min Inn", min_value=0, max_value=20000, value=200, key=f"{prefix}_min_inn_start")
            st.number_input("End Year Min Inn",   min_value=0, max_value=20000, value=200, key=f"{prefix}_min_inn_end")
        else:
            st.number_input("Start Year Min PA", min_value=0, max_value=20000, value=min_pa_start, key=f"{prefix}_min_pa_start")
            st.number_input("End Year Min PA",   min_value=0, max_value=20000, value=min_pa_end,   key=f"{prefix}_min_pa_end")
    else:
        st.number_input("Start Year min IP", min_value=0, max_value=5000, value=min_ip_start, key=f"{prefix}_min_ip_start")
        st.number_input("End Year min IP", min_value=0, max_value=5000, value=min_ip_end, key=f"{prefix}_min_ip_end")

    if is_hitting:
        st.selectbox("Position", options=list(POSITION_OPTIONS.keys()),
                 format_func=lambda x: POSITION_OPTIONS[x], key=f"{prefix}_position")
    st.selectbox("Team", options=list(TEAM_OPTIONS.keys()),
                 format_func=lambda x: TEAM_OPTIONS[x], key=f"{prefix}_team",
                 help="Filters by team in the end year only")

    min_pa_start_val  = int(st.session_state.get(f"{prefix}_min_pa_start", 0))
    min_pa_end_val    = int(st.session_state.get(f"{prefix}_min_pa_end",   0))
    min_inn_start_val = int(st.session_state.get(f"{prefix}_min_inn_start", 0))
    min_inn_end_val   = int(st.session_state.get(f"{prefix}_min_inn_end",   0))
    min_ip_start_val   = int(st.session_state.get(f"{prefix}_min_ip_start", 0))
    min_ip_end_val   = int(st.session_state.get(f"{prefix}_min_ip_end", 0))
    position_val = st.session_state.get(f"{prefix}_position", "all")
    team_val     = st.session_state.get(f"{prefix}_team", "all")

    st.checkbox("Show Decliners", key=f"{prefix}_show_fallers")
    if is_hitting:
        if use_inn:
            st.checkbox("Show Min Inn",    key=f"{prefix}_show_min_pa")
            st.checkbox("Show Player Inn", key=f"{prefix}_show_player_pa")
        else:
            st.checkbox("Show Min PA",    key=f"{prefix}_show_min_pa")
            st.checkbox("Show Player PA", key=f"{prefix}_show_player_pa")
    else:
        st.checkbox("Show min IP",     key=f"{prefix}_show_min_ip")
        st.checkbox("Show player IP",  key=f"{prefix}_show_player_ip")


show_fallers = st.session_state.get(f"{prefix}_show_fallers", False)

# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────

if end_year > start_year:
    with st.spinner("Loading data..."):
        if is_hitting:
            df = load_risers_data(
                start_year, end_year,
                min_pa_start=min_pa_start_val, min_pa_end=min_pa_end_val,
                min_inn_start=min_inn_start_val, min_inn_end=min_inn_end_val,
                use_inn=use_inn,
                position=position_val, team=team_val,
            )
        else:
            df = load_risers_data(start_year, end_year, min_ip_start_val, min_ip_end_val, team_val)
else:
    df = pd.DataFrame()

# Sort & filter direction
if not df.empty and stat in df.columns:
    stat_lower = stat in lower_better
    ascending  = (stat_lower and not show_fallers) or (not stat_lower and show_fallers)
    df = df.sort_values(by=stat, ascending=ascending)
    numeric_delta = pd.to_numeric(df[stat], errors="coerce")
    if stat_lower:
        df = df[numeric_delta < 0] if not show_fallers else df[numeric_delta > 0]
    else:
        df = df[numeric_delta > 0] if not show_fallers else df[numeric_delta < 0]
    df = df.head(10)
elif not df.empty:
    st.error(f"Stat '{stat}' not found in dataset.")
    df = pd.DataFrame()

# ─────────────────────────────────────────────
#  Build cards
# ─────────────────────────────────────────────

cards = []
for _, row in df.iterrows():
    name  = str(row.get("Name", "")).strip()
    team  = str(row.get("Team", ""))
    delta = row.get(stat, np.nan)

    is_positive = pd.notna(delta) and float(delta) > 0
    display_val = format_stat_yoy(stat, delta, show_sign=is_positive)

    end_val     = row.get(f"{stat}_end", np.nan)
    end_display = format_stat_yoy(stat, end_val) if pd.notna(end_val) else ""
    stat_label  = label_map.get(stat, stat)

    player_pa_html = ""
    if is_hitting:
        if st.session_state.get(f"{prefix}_show_player_pa"):
            if use_inn:
                inn_s = row.get("Inn_start", np.nan)
                inn_e = row.get("Inn_end",   np.nan)
                parts = []
                if pd.notna(inn_s): parts.append(f"{inn_s:.1f}".rstrip("0").rstrip(".") if inn_s != int(inn_s) else str(int(inn_s)))
                if pd.notna(inn_e): parts.append(f"{inn_e:.1f}".rstrip("0").rstrip(".") if inn_e != int(inn_e) else str(int(inn_e)))
                if parts:
                    player_pa_html = f'<div class="player-pa">{" → ".join(parts)} Inn</div>'
            else:
                pa_start = row.get("PA_start", np.nan)
                pa_end   = row.get("PA_end",   np.nan)
                parts = []
                if pd.notna(pa_start): parts.append(str(int(pa_start)))
                if pd.notna(pa_end):   parts.append(str(int(pa_end)))
                if parts:
                    player_pa_html = f'<div class="player-pa">{" → ".join(parts)} PA</div>'
    else:
        ip_start = row.get("IP_start", np.nan)
        ip_end   = row.get("IP_end",   np.nan)
        if st.session_state.get(f"{prefix}_show_player_ip"):
            parts = []
            if pd.notna(ip_start): parts.append(format_stat_yoy("IP", ip_start))
            if pd.notna(ip_end):   parts.append(format_stat_yoy("IP", ip_end))
            if parts:
                player_pa_html = f'<div class="player-pa">{" → ".join(parts)} IP</div>'

    if stat in lower_better:
        is_improvement = pd.notna(delta) and float(delta) < 0
    else:
        is_improvement = is_positive
    delta_class = "stat-positive" if is_improvement else "stat-negative"
    end_context = f'<div class="player-endval">{stat_label}: {end_display}</div>' if end_display else ""

    src      = get_headshot(row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}"/>'
    cards.append(f"""
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(name)}</div>
      <div class="player-team">{html.escape(team)}</div>
      <div class="player-stat {delta_class}">{html.escape(display_val)}</div>
      {end_context}
      {player_pa_html}
    </div>
    """)

# ─────────────────────────────────────────────
#  Title & subtitle
# ─────────────────────────────────────────────

title_stat_label = label_map.get(stat, stat)
pos_suffix  = f" ({POSITION_OPTIONS.get(position_val, '')})" if position_val != "all" else ""
team_prefix = f"{TEAM_OPTIONS.get(team_val, '')} " if team_val != "all" else ""
riser_label = "Decliners" if show_fallers else "Improvers"
title = f"Top {team_prefix}{title_stat_label} {riser_label}{pos_suffix}: {int(start_year)} → {int(end_year)}"

min_pa_subtitle = ""
if is_hitting:
    if st.session_state.get(f"{prefix}_show_min_pa"):
        if use_inn:
            min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_inn_start_val} Inn → {min_inn_end_val} Inn</div>'
        else:
            min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_pa_start_val} PA → {min_pa_end_val} PA</div>'
else:
    if st.session_state.get(f"{prefix}_show_min_ip"):
        min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_ip_start_val} IP → {min_ip_end_val} IP</div>'


# ─────────────────────────────────────────────
#  Render
# ─────────────────────────────────────────────

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_pa_subtitle}
    <div class="players-grid">
        {''.join(cards) if cards else '<div style="padding:2rem;color:#999;">No data found. Try adjusting your filters or years.</div>'}
    </div>
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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
.leaderboard-card {{
    background: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto;
    width: 100%;
    max-width: 900px;
    box-sizing: border-box;
    font-family: "Source Sans Pro", sans-serif;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.2rem;
    margin-bottom: 1rem;
    text-align: center;
}}
.leaderboard-subtitle {{
    text-align: center;
    color: #888;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    margin-top: -0.5rem;
}}
.players-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2.5rem 1rem;
}}
.player-card {{
    flex: 0 0 145px;
    width: 145px;
    text-align: center;
}}
.player-card img {{
    width: 145px;
    height: 145px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    background: #f6f6f6;
}}
.player-name {{ font-weight: 800; margin-top: 0.35rem; font-size: 1.1rem; }}
.player-team {{ color: #666; font-size: 0.85rem; }}
.player-stat {{ font-weight: 900; font-size: 1.5rem; margin-top: 0.25rem; }}
.stat-positive {{ color: #1a7a3c; }}
.stat-negative {{ color: #c0392b; }}
.player-endval {{ color: #888; font-size: .9rem; margin-top: 0.1rem; }}
.player-pa {{ color: #666; font-size: 0.9rem; margin-top: 0.1rem; }}
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

/* Compact Screenshot Overrides for Mobile Screens */
@media (max-width: 600px) {{
    .leaderboard-card {{
        width: 100% !important;
        padding: 1.5rem 0.5rem;
    }}
    .leaderboard-title {{
        font-size: 1.4rem;
        margin-bottom: 0.6rem;
    }}
    .leaderboard-subtitle {{
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }}
    .players-grid {{
        gap: 1rem 0.35rem; /* Tighter item gutters */
    }}
    .player-card {{
        flex: 0 0 calc(20% - 0.3rem); /* Forces the clean 5x2 rows */
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
    .player-stat {{ font-size: 0.9rem; }}
    .player-endval {{ font-size: 0.6rem; }}
    .player-pa {{ font-size: 0.55rem; }}
    .footer {{ 
        padding: 0 0.5rem; 
        margin-top: 1rem;
    }}
       .footer p {{
        font-size: 0.65rem;
    }}

    .footer {{
    padding: 0 2rem;
    }}
    .leaderboard-subtitle {{
    margin-top: 0rem;
}}
}}
</style>
</head>
<body>{grid_html}</body>
</html>
"""

with col2:
    if view_mode == "Graphic":
        st.iframe(full_html, height=850)

    else:
        if is_hitting:
            df_full = load_risers_data(
                start_year, end_year,
                min_pa_start=min_pa_start_val, min_pa_end=min_pa_end_val,
                min_inn_start=min_inn_start_val, min_inn_end=min_inn_end_val,
                use_inn=use_inn,
                position=position_val, team=team_val,
            )
        else:
            df_full = df = load_risers_data(start_year, end_year, min_ip_start_val, min_ip_end_val, team_val)

        if df_full.empty or stat not in df_full.columns:
            st.info("No data found. Try adjusting your filters or years.")
            st.stop()

        stat_lower = stat in lower_better
        df_full = df_full.sort_values(by=stat, ascending=stat_lower)

        stat_label = label_map.get(stat, stat)
        start_col  = f"{stat}_start"
        end_col    = f"{stat}_end"
        col_start  = f"{stat_label} ({start_year})"
        col_end    = f"{stat_label} ({end_year})"
        col_delta  = f"Change in {stat_label}"

        base_cols = [c for c in ["Name", "Team"] if c in df_full.columns]
        display = df_full[base_cols + [start_col, end_col, stat]].copy()
        display = display.rename(columns={
            start_col: col_start,
            end_col:   col_end,
            stat:      col_delta,
        })
        display = display.reset_index(drop=True)
        display.index += 1

        decimals = STAT_ROUND.get(stat, 1)
        fmt = f"%.{decimals}f"
        col_config = {
            col_start: st.column_config.NumberColumn(col_start, format=fmt),
            col_end:   st.column_config.NumberColumn(col_end,   format=fmt),
            col_delta: st.column_config.NumberColumn(col_delta, format=fmt),
        }

        st.caption(f"{stat_label} — {int(start_year)} → {int(end_year)}")
        st.dataframe(display, width="stretch", height=700, column_config=col_config)
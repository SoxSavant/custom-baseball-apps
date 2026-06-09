import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import html
from datetime import date

st.set_page_config(page_title="Hitter Season Counter", layout="wide", page_icon="⚾")

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
    st.title("Hitter Season Counter")
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
from h_utils import (
    STAT_ALLOWLIST, RATE_STATS, format_stat, STAT_DEFAULTS, STAT_ROUND, normalize_team,
    get_headshot, label_map, lower_better, start_year,
    POSITION_OPTIONS, get_team_display, filter_by_position, load_final_year
)

from utils import TEAM_OPTIONS

MAX_DISPLAY  = 10
current_year = date.today().year

def update_stat_default(i):
    stat = st.session_state[f"hs_stat_{i}"]
    st.session_state[f"hs_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"hs_op_{i}"]  = "<=" if stat in lower_better else ">="


def load_all_seasons(start: int, end: int) -> pd.DataFrame:
    frames = []
    for year in range(start, end + 1):
        df = load_final_year(year)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

for key, default in [
    ("hs_start_year",  current_year - 10),
    ("hs_end_year",    current_year-1),
    ("hs_position",    "all"),
    ("hs_min_pa", 300),
    ("hs_min_type","PA"),
    ("hs_show_min_pa", False),
    ("hs_val_0",       5.0),
    ("hs_val_1",       100),
    ("hs_view", "Graphic"),
    ("hs_team", "all"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([0.5, 2])

with col1:
    view_mode = st.radio("View", ["Graphic", "Database"], key="hs_view", horizontal=True)
    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=0, horizontal=True, key="hs_num_stats")

    st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key="hs_start_year")
    st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key="hs_end_year")

    sel_start = st.session_state["hs_start_year"]
    sel_end   = max(st.session_state["hs_end_year"], sel_start)

    st.selectbox("Min Type (per season)", options=["PA", "Inn"], key="hs_min_type")

    if st.session_state["hs_min_type"] == "Inn":
        st.number_input("Min Inn", min_value=0, max_value=20000, value=200, key="hs_min_inn")
    else:
        st.number_input("Min PA", min_value=0, max_value=20000, key="hs_min_pa")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        default_stat  = "fWAR" if i == 0 else "RBI" if i == 1 else STAT_ALLOWLIST[0]

        chosen_stat = st.selectbox(
            f"Stat {i+1}", STAT_ALLOWLIST,
            key=f"hs_stat_{i}",
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed",
            on_change=update_stat_default,
            args=(i,),
        )

        op_col, val_col = st.columns([1, 2])
        with op_col:
            st.selectbox("Op", [">=", "<="], key=f"hs_op_{i}", label_visibility="collapsed")
        with val_col:
            decimals = STAT_ROUND.get(chosen_stat, 0)
            step = 10 ** -decimals if decimals > 0 else 1.0
            fmt = f"%.{decimals}f"
            st.number_input(
                f"Value {i+1}", step=step, key=f"hs_val_{i}",
                label_visibility="collapsed", format=fmt,
            )

    st.selectbox(
        "Position", options=list(POSITION_OPTIONS.keys()),
        format_func=lambda x: POSITION_OPTIONS[x], key="hs_position",
    )

    st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key="hs_team",
    )

    if st.session_state["hs_min_type"] == "PA":
        st.checkbox("Show Min PA", key="hs_show_min_pa")
    else:
        st.checkbox("Show Min Inn", key="hs_show_min_pa")

active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"hs_stat_{i}")
    op   = st.session_state.get(f"hs_op_{i}", ">=")
    val  = float(st.session_state.get(f"hs_val_{i}", 0.0))
    if stat:
        active_filters.append((stat, op, val))

use_inn     = st.session_state.get("hs_min_type") == "Inn"
min_pa_val  = int(st.session_state.get("hs_min_pa", 0))
min_inn_val = int(st.session_state.get("hs_min_inn", 0))
position = st.session_state["hs_position"]
team_val = st.session_state.get("hs_team", "all")

df_all = load_all_seasons(sel_start, sel_end)

if df_all is None or df_all.empty:
    st.error(f"No data found for {sel_start}–{sel_end}.")
    st.stop()

if use_inn:
    if min_inn_val > 0 and "Inn" in df_all.columns:
        df_all = df_all[pd.to_numeric(df_all["Inn"], errors="coerce").fillna(0) >= min_inn_val]
else:
    if min_pa_val > 0 and "PA" in df_all.columns:
        df_all = df_all[pd.to_numeric(df_all["PA"], errors="coerce").fillna(0) >= min_pa_val]

if team_val != "all" and "Team" in df_all.columns:
    target = normalize_team(team_val)
    df_all = df_all[df_all["Team"].astype(str).apply(lambda t: normalize_team(t) == target)]

# Position filter
df_all = filter_by_position(df_all, position)

# Apply all stat filters — a season row must pass ALL filters
mask = pd.Series([True] * len(df_all), index=df_all.index)
for stat, op, val in active_filters:
    if stat not in df_all.columns:
        continue
    col_vals    = pd.to_numeric(df_all[stat], errors="coerce")
    decimals = STAT_ROUND.get(stat, 0)
    col_vals = col_vals.round(decimals)
    compare_val = val
    if stat in RATE_STATS:
        median_col = col_vals.median()
        if pd.notna(median_col) and median_col <= 1 and val > 1:
            compare_val = val / 100
    if op == ">=":
        mask = mask & (col_vals >= compare_val)
    else:
        mask = mask & (col_vals <= compare_val)

qualifying_rows = df_all[mask]


if qualifying_rows.empty:
    display_df = pd.DataFrame()
    display_df_graphic = pd.DataFrame()
else:
    # 1. Group to get counts, list of seasons, and count of unique teams
    grouped = (
        qualifying_rows
        .groupby("PlayerId")
        .agg(
            season_count=("Season", "count"),
            seasons=("Season", lambda x: sorted(x.tolist())),
            unique_team_count=("Team", "nunique")
        )
        .reset_index()
    )
    
    # 2. Get the most recent team for the display (fallback)
    last_team_meta = (
        qualifying_rows
        .sort_values("Season")
        .groupby("PlayerId")
        .last()
        .reset_index()
    )
    
    # 3. Merge them
    display_df = grouped.merge(last_team_meta, on="PlayerId", how="left")
    
    # 4. Apply if unique teams > 1, set Team to "2+ Teams"
    display_df["Team"] = display_df.apply(
        lambda x: "2+ Teams" if x["unique_team_count"] > 1 else x["Team"], 
        axis=1
    )
    
    # 5. Sort by season count for the leaderboard
    display_df = display_df.sort_values("season_count", ascending=False)
    display_df_graphic = display_df.head(MAX_DISPLAY)

def format_threshold_label(stat, val, op):
    lbl       = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


cards = []
for _, row in display_df_graphic.iterrows():
    name     = str(row.get("Name", "")).strip()
    team_raw = str(row.get("Team", ""))
    team     = get_team_display(team_raw)

    season_count = int(row.get("season_count", 0))
    seasons_list = row.get("seasons", [])
    seasons_str  = ", ".join(str(y) for y in seasons_list)
    season_word  = "Season" if season_count == 1 else "Seasons"

    src      = get_headshot(row)
    img_html = (
        f'<img src="{html.escape(src)}" alt="{html.escape(name)}" '
        f' style="object-fit:cover;border-radius:6px;'
        f'border:1px solid #e0e0e0;background:#f6f6f6;display:block;"/>'
    )

    cards.append(f"""
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(name)}</div>
      <div class="player-team">{html.escape(team)}</div>
      <div class="season-count">{season_count} {season_word}</div>
      <div class="season-years">{html.escape(seasons_str)}</div>
    </div>""")

filter_parts  = [format_threshold_label(s, v, op) for s, op, v in active_filters]
threshold_str = ", ".join(filter_parts)
span_label    = f"{sel_start}–{sel_end}"
pos_val       = st.session_state["hs_position"]
pos_suffix    = f" ({POSITION_OPTIONS[pos_val]})" if pos_val != "all" else ""
team_label  = f" ({TEAM_OPTIONS.get(team_val, "")})" if team_val != "all" else ""

page_title = f"Most {threshold_str} Seasons: {span_label}{pos_suffix} {team_label}"

if st.session_state.get("hs_show_min_pa"):
    if use_inn:
        min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_inn_val} Inn per season</div>'
    else:
        min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_pa_val} PA per season</div>'
else:
    min_pa_subtitle = ""

body = "".join(cards) if cards else (
    '<div style="padding:2rem;color:#999;text-align:center;">'
    'No players matched the filter. Try adjusting your threshold.</div>'
)

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(page_title)}</div>
    {min_pa_subtitle}
    <div class="players-grid">{body}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p>Data: FanGraphs • Baseball Reference • Baseball Savant</p>
    </div>
</div>
"""

card_count = len(cards)
est_rows   = max(1, (card_count + 4) // 5)
est_height = 180 + est_rows * 310 + 80

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
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.25rem;
    margin-bottom: 1.2rem;
    text-align: center;
    line-height: 1.2;
}}
.leaderboard-subtitle {{
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    margin-top: -0.5rem;
}}

/* FIXED: Changed from grid to flex framework to handle centered trailing items */
.players-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2rem 1rem;
}}

/* FIXED: Explicit card baseline sizing for centered desktop flow */
.player-card {{ 
    flex: 0 0 155px;
    width: 155px; 
    text-align: center; 
    min-width: 0;
}}
.player-card img {{
    width: 100%; 
    max-width: 155px;
    aspect-ratio: 1 / 1; 
    object-fit: cover; 
    border-radius: 6px;
    border: 1px solid #e0e0e0; 
    background: #f6f6f6;
    display: block;
    margin: 0 auto;
}}
.player-name {{ font-weight: 800; font-size: 1rem; margin-top: 0.35rem; line-height: 1.2; }}
.player-team {{ color: #666; font-size: 0.8rem; margin-bottom: 0.2rem; }}
.season-count {{ font-weight: 800; font-size: 1.05rem; color: #1a1a1a; margin-top: 0.25rem; }}
.season-years {{ color: #aaa; font-size: 0.78rem; margin-top: 0.1rem; line-height: 1.3; }}

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
        gap: 1rem 0.35rem; /* Tighter column gutters */
    }}
    
    /* FIXED: Forces clean 5-across columns scaling dynamically on phone frames */
    .player-card {{
        flex: 0 0 calc(20% - 0.3rem);
        width: calc(20% - 0.3rem);
    }}

    .player-name {{ 
        font-size: 0.65rem; 
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis; 
    }}
    .player-team {{ font-size: 0.55rem; }}
    .season-count {{ font-size: 0.75rem; }}
    .season-years {{ font-size: 0.55rem; }}

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
    if view_mode == "Graphic":
        components.html(full_html, height=est_height, scrolling=True)

    else:
        if display_df.empty:
            st.info("No players matched the filter. Try adjusting your threshold.")
            st.stop()

        rows = []
        for _, row in display_df.iterrows():
            name         = str(row.get("Name", "")).strip()
            team_raw     = str(row.get("Team", ""))
            team         = get_team_display(team_raw)
            season_count = int(row.get("season_count", 0))
            seasons_list = row.get("seasons", [])
            seasons_str  = ", ".join(str(y) for y in seasons_list)
            rows.append({
                "Name":     name,
                "Team":     team,
                "Seasons":  season_count,
                "Years":    seasons_str,
            })

        db_df = pd.DataFrame(rows)
        db_df.index += 1

        total = len(db_df)
        st.caption(f"{page_title}")
        st.dataframe(
            db_df,
            width="stretch",
            height=700,
            column_config={
                "Name":    st.column_config.TextColumn("Name",    width="medium"),
                "Team":    st.column_config.TextColumn("Team",    width="small"),
                "Seasons": st.column_config.NumberColumn("Seasons", format="%d", width="small"),
                "Years":   st.column_config.TextColumn("Years",   width="large"),
            },
        )

        st.markdown(
            "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem;'>"
            "Data: Baseball Reference · FanGraphs · Baseball Savant"
            "</div>",
            unsafe_allow_html=True,
        )
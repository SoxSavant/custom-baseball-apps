import streamlit as st
import pandas as pd
import numpy as np
import html
from datetime import date
import h_utils
import p_utils
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import unicodedata


st.set_page_config(page_title="Season Counter", layout="wide", page_icon="⚾")

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
    st.title("Season Counter")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

type_mode = st.radio("Type", ["Hitting", "Pitching","Combined"], horizontal=True, key="cc_mode", label_visibility="collapsed")
is_hitting = (type_mode == "Hitting")
is_pitching = (type_mode == "Pitching")
is_combined = (type_mode == "Combined")

# Combined mode reuses h_utils for display helpers (headshots, team display,
# formatting) — same approach as war_leaders_app.py.
U = p_utils if is_pitching else h_utils
prefix = "hcc" if is_hitting else "pcc" if is_pitching else "ccc"

current_year = date.today().year
last_updated = U.get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")


from h_utils import (
    STAT_ALLOWLIST, normalize_team,
    POSITION_OPTIONS, filter_by_position
)

from utils import TEAM_OPTIONS

WAR_STATS = ["fWAR", "bWAR", "fWAR-bWAR Avg"]

STAT_ALLOWLIST = WAR_STATS if is_combined else U.STAT_ALLOWLIST
RATE_STATS = U.RATE_STATS
STAT_ROUND = U.STAT_ROUND
STAT_DEFAULTS = U.STAT_DEFAULTS
label_map = U.label_map
lower_better = U.lower_better
start_year = U.start_year
format_stat = U.format_stat
load_final_year = U.load_final_year
STAT_DISPLAY_NAMES = U.STAT_DISPLAY_NAMES
get_team_display = U.get_team_display
get_headshot = U.get_headshot

MAX_DISPLAY  = 10

SLIM_COLS = ["PlayerId", "Name", "Team", "MLBAMID", "fWAR", "bWAR"]


def _slim(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Pull PlayerId/Name/Team/MLBAMID/fWAR/bWAR out of a hitting or pitching
    frame, renaming the movable columns with a suffix so they survive a merge."""
    out = pd.DataFrame(columns=SLIM_COLS)
    if df is not None and not df.empty:
        present = [c for c in SLIM_COLS if c in df.columns]
        out = df[present].copy()
        for c in SLIM_COLS:
            if c not in out.columns:
                out[c] = np.nan
        out["PlayerId"] = pd.to_numeric(out["PlayerId"], errors="coerce").astype("Int64").astype(str)
    return out.rename(columns={
        "Name": f"Name_{suffix}",
        "Team": f"Team_{suffix}",
        "MLBAMID": f"MLBAMID_{suffix}",
        "fWAR": f"fWAR_{suffix}",
        "bWAR": f"bWAR_{suffix}",
    })


def load_combined_year(year: int) -> pd.DataFrame:
    hit_slim = _slim(h_utils.load_final_year(year), "h")
    pit_slim = _slim(p_utils.load_final_year(year), "p")

    merged = hit_slim.merge(pit_slim, on="PlayerId", how="outer")

    merged["Name"] = merged["Name_h"].fillna(merged["Name_p"])
    merged["Team"] = merged["Team_h"].fillna(merged["Team_p"])
    merged["MLBAMID"] = merged["MLBAMID_h"].fillna(merged["MLBAMID_p"])

    merged["fWAR"] = pd.to_numeric(merged["fWAR_h"], errors="coerce").fillna(0) + \
                      pd.to_numeric(merged["fWAR_p"], errors="coerce").fillna(0)
    merged["bWAR"] = pd.to_numeric(merged["bWAR_h"], errors="coerce").fillna(0) + \
                      pd.to_numeric(merged["bWAR_p"], errors="coerce").fillna(0)
    merged["fWAR-bWAR Avg"] = (merged["fWAR"] + merged["bWAR"]) / 2

    merged = merged.drop(columns=[
        "Name_h", "Name_p", "Team_h", "Team_p",
        "MLBAMID_h", "MLBAMID_p", "fWAR_h", "fWAR_p", "bWAR_h", "bWAR_p",
    ])
    merged = merged[merged["PlayerId"].notna() & (merged["PlayerId"] != "<NA>")]
    merged = merged[merged["Name"].notna()]
    return merged


def update_stat_default(i):
    stat = st.session_state[f"{prefix}_stat_{i}"]
    st.session_state[f"{prefix}_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"{prefix}_op_{i}"]  = "<=" if stat in lower_better else ">="


def load_all_seasons(start: int, end: int) -> pd.DataFrame:
    frames = []
    for year in range(start, end + 1):
        if is_combined:
            df = load_combined_year(year)
            if df is not None and not df.empty:
                df = df.copy()
                df["Season"] = year
        else:
            df = load_final_year(year)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

for key, default in [
    (f"{prefix}_start_year",  current_year - 10),
    (f"{prefix}_end_year",    current_year-1),
    (f"{prefix}_position",    "all"),
    (f"{prefix}_min_pa", 300),
    (f"{prefix}_min_ip", 100),
    (f"{prefix}_min_type","PA"),
    (f"{prefix}_show_min_pa", False),
    (f"{prefix}_val_0",       5.0),
    (f"{prefix}_val_1",       5.0),
    (f"{prefix}_view", "Graphic"),
    (f"{prefix}_team", "all"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([0.5, 2])

with col1:
    view_mode = st.radio("View", ["Graphic", "Database"], key=f"{prefix}_view", horizontal=True)

    num_stats_options = [1, 2, 3] if is_combined else [1, 2, 3, 4]
    num_stats = st.radio("Number of stat filters", num_stats_options, index=0, horizontal=True, key=f"{prefix}_num_stats")

    st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key=f"{prefix}_start_year")
    st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key=f"{prefix}_end_year")

    sel_start = st.session_state[f"{prefix}_start_year"]
    sel_end   = max(st.session_state[f"{prefix}_end_year"], sel_start)

    if is_hitting:
        st.selectbox("Min Type", options=["PA", "Inn"], key=f"{prefix}_min_type")
        if st.session_state[f"{prefix}_min_type"] == "Inn":
            st.number_input("Min Inn", min_value=0, max_value=20000, value=200, key=f"{prefix}_min_inn")
        else:
            st.number_input("Min PA (per season)", min_value=0, max_value=20000, key=f"{prefix}_min_pa")
    elif is_pitching:
        st.number_input("Min IP (per season)", min_value=0, max_value=5000, key=f"{prefix}_min_ip")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        if is_combined:
            default_stat = WAR_STATS[i] if i < len(WAR_STATS) else WAR_STATS[0]
        else:
            default_stat = "fWAR" if i == 0 else "bWAR" if i == 1 else STAT_ALLOWLIST[0]

        chosen_stat = st.selectbox(
            f"Stat {i+1}", STAT_ALLOWLIST,
            key=f"{prefix}_stat_{i}",
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed",
            on_change=update_stat_default,
            args=(i,),
        )

        op_col, val_col = st.columns([1, 2])
        with op_col:
            st.selectbox("Op", [">=", "<="], key=f"{prefix}_op_{i}", label_visibility="collapsed")
        with val_col:
            decimals = 1 if is_combined else STAT_ROUND.get(chosen_stat, 0)
            step = 10 ** -decimals if decimals > 0 else 1.0
            fmt = f"%.{decimals}f"
            st.number_input(
                f"Value {i+1}", step=step, key=f"{prefix}_val_{i}",
                label_visibility="collapsed", format=fmt,
            )

    if is_hitting:
        st.selectbox(
            "Position", options=list(POSITION_OPTIONS.keys()),
            format_func=lambda x: POSITION_OPTIONS[x], key=f"{prefix}_position",
        )

    st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key=f"{prefix}_team",
    )

    if is_hitting:
        if st.session_state[f"{prefix}_min_type"] == "PA":
            st.checkbox("Show Min PA", key=f"{prefix}_show_min_pa")
        else:
            st.checkbox("Show Min Inn", key=f"{prefix}_show_min_pa")
    elif is_pitching:
        st.checkbox("Show Min IP", key = f"{prefix}_show_min_ip")
    # Combined: nothing to show/hide here.

active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"{prefix}_stat_{i}")
    op   = st.session_state.get(f"{prefix}_op_{i}", ">=")
    val  = float(st.session_state.get(f"{prefix}_val_{i}", 0.0))
    if stat:
        active_filters.append((stat, op, val))

use_inn     = st.session_state.get(f"{prefix}_min_type") == "Inn"
min_pa_val  = int(st.session_state.get(f"{prefix}_min_pa", 0))
min_inn_val = int(st.session_state.get(f"{prefix}_min_inn", 0))
min_ip_val = int(st.session_state.get(f"{prefix}_min_ip", 0))
position = st.session_state[f"{prefix}_position"]
team_val = st.session_state.get(f"{prefix}_team", "all")

df_all = load_all_seasons(sel_start, sel_end)

if df_all is None or df_all.empty:
    st.error(f"No data found for {sel_start}–{sel_end}.")
    st.stop()

if is_hitting:
    if use_inn:
        if min_inn_val > 0 and "Inn" in df_all.columns:
            df_all = df_all[pd.to_numeric(df_all["Inn"], errors="coerce").fillna(0) >= min_inn_val]
    else:
        if min_pa_val > 0 and "PA" in df_all.columns:
            df_all = df_all[pd.to_numeric(df_all["PA"], errors="coerce").fillna(0) >= min_pa_val]
elif is_pitching:
    if min_ip_val > 0 and "IP" in df_all.columns:
        df_all = df_all[pd.to_numeric(df_all["IP"], errors="coerce").fillna(0) >= min_ip_val]
# Combined: no PA/IP threshold applied.

if team_val != "all" and "Team" in df_all.columns:
    target = normalize_team(team_val)
    df_all = df_all[df_all["Team"].astype(str).apply(lambda t: normalize_team(t) == target)]

if is_hitting:
    df_all = filter_by_position(df_all, position)

# Apply all stat filters — a season row must pass ALL filters
mask = pd.Series([True] * len(df_all), index=df_all.index)
for stat, op, val in active_filters:
    if stat not in df_all.columns:
        continue
    col_vals    = pd.to_numeric(df_all[stat], errors="coerce")
    decimals = 1 if is_combined else STAT_ROUND.get(stat, 0)
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
pos_val       = st.session_state[f"{prefix}_position"]
pos_suffix    = f" ({POSITION_OPTIONS[pos_val]})" if pos_val != "all" else ""
team_label = f' ({TEAM_OPTIONS.get(team_val, "")})' if team_val != "all" else ""

page_title = f"Most {threshold_str} Seasons: {span_label}{pos_suffix} {team_label}"

if is_hitting:
    if st.session_state.get(f"{prefix}_show_min_pa"):
        if use_inn:
            min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_inn_val} Inn per season</div>'
        else:
            min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_pa_val} PA per season</div>'
    else:
        min_pa_subtitle = ""
elif is_pitching:
    if st.session_state.get(f"{prefix}_show_min_ip"):
        min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {min_ip_val} IP per season</div>'
    else:
        min_pa_subtitle = ""
else:
    min_pa_subtitle = '<div class="leaderboard-subtitle">Hitting + Pitching</div>'
 


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
        st.iframe(full_html, height=est_height)

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
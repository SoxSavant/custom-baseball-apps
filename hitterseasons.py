import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import html
from datetime import date

st.set_page_config(page_title="Hitter Season Counter Leaderboard", layout="wide", page_icon="⚾")

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
    st.title("Hitter Season Counter Leaderboard")
with meta_col:
    st.markdown(
        '<div style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Imports from shared utils
# ─────────────────────────────────────────────

from h_utils import (
    STAT_ALLOWLIST, RATE_STATS, format_stat, STAT_DEFAULTS,
    get_headshot, label_map, lower_better, start_year,
    POSITION_OPTIONS, get_team_display,
)

MAX_DISPLAY  = 10
current_year = date.today().year

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def apply_dh_override(df: pd.DataFrame) -> pd.DataFrame:
    if "Pos" not in df.columns or "PA" not in df.columns or "Inn" not in df.columns:
        return df
    df = df.copy()
    pa  = pd.to_numeric(df["PA"],  errors="coerce").fillna(0)
    inn = pd.to_numeric(df["Inn"], errors="coerce").fillna(0)
    estimated = (pa / 4.1) * 9
    is_dh = (inn == 0) | ((inn > 0) & (estimated / inn > 3))
    df.loc[is_dh, "Pos"] = "DH"
    return df


def filter_by_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    df = apply_dh_override(df)
    if position == "all" or "Pos" not in df.columns:
        return df
    if position == "OF":
        return df[df["Pos"].astype(str).str.upper().isin(["LF", "CF", "RF"])]
    return df[df["Pos"].astype(str).str.upper() == position.upper()]


def update_stat_default(i):
    stat = st.session_state[f"hs_stat_{i}"]
    st.session_state[f"hs_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"hs_op_{i}"]  = "<=" if stat in lower_better else ">="


# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_year(year: int) -> pd.DataFrame:
    path = f"data/final/hitting_final_{year}.csv"
    try:
        df = pd.read_csv(path)
        df["Season"] = year
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_all_seasons(start: int, end: int) -> pd.DataFrame:
    frames = []
    for year in range(start, end + 1):
        df = load_final_year(year)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
#  Session state defaults (no widget keys here)
# ─────────────────────────────────────────────

for key, default in [
    ("hs_start_year",  current_year - 10),
    ("hs_end_year",    current_year-1),
    ("hs_position",    "all"),
    ("hs_min_pa",      300),
    ("hs_show_min_pa", False),
    ("hs_val_0",       5.0),
    ("hs_val_1",       100),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Controls
# ─────────────────────────────────────────────

col1, col2 = st.columns([0.5, 2])

with col1:
    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=0, horizontal=True, key="hs_num_stats")

    st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key="hs_start_year")
    st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key="hs_end_year")

    sel_start = st.session_state["hs_start_year"]
    sel_end   = max(st.session_state["hs_end_year"], sel_start)

    st.number_input("Min PA (per season)", min_value=0, max_value=5000, key="hs_min_pa")

    RATE_STATS_3DP = {"AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "ISO", "BABIP"}

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
            if chosen_stat in RATE_STATS_3DP or chosen_stat == "wOBA-xwOBA":
                step, fmt = 0.001, "%.3f"
            elif "%" in chosen_stat or  chosen_stat in {"EV", "fWAR", "bWAR", "BatSpd", "Def" ,"Off"}:
                step, fmt = 0.1, "%.1f"
            elif chosen_stat in {"WPA", "Clutch"}:
                step, fmt = 0.01, "%.2f"
            else:
                step, fmt = 1.0, "%.0f"
            st.number_input(f"Value {i+1}", step=step, key=f"hs_val_{i}",
                            label_visibility="collapsed", format=fmt)

    st.selectbox(
        "Position", options=list(POSITION_OPTIONS.keys()),
        format_func=lambda x: POSITION_OPTIONS[x], key="hs_position",
    )

    st.checkbox("Show min PA", key="hs_show_min_pa")

# ─────────────────────────────────────────────
#  Collect active filters
# ─────────────────────────────────────────────

active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"hs_stat_{i}")
    op   = st.session_state.get(f"hs_op_{i}", ">=")
    val  = float(st.session_state.get(f"hs_val_{i}", 0.0))
    if stat:
        active_filters.append((stat, op, val))

min_pa   = int(st.session_state["hs_min_pa"])
position = st.session_state["hs_position"]

# ─────────────────────────────────────────────
#  Load & filter per-season rows
# ─────────────────────────────────────────────

df_all = load_all_seasons(sel_start, sel_end)

if df_all is None or df_all.empty:
    st.error(f"No data found for {sel_start}–{sel_end}.")
    st.stop()


# Min PA filter per season row
if min_pa > 0 and "PA" in df_all.columns:
    df_all = df_all[pd.to_numeric(df_all["PA"], errors="coerce").fillna(0) >= min_pa]

# Position filter
df_all = filter_by_position(df_all, position)

# Apply all stat filters — a season row must pass ALL filters
mask = pd.Series([True] * len(df_all), index=df_all.index)
for stat, op, val in active_filters:
    if stat not in df_all.columns:
        continue
    col_vals    = pd.to_numeric(df_all[stat], errors="coerce")
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


# ─────────────────────────────────────────────
#  Aggregate: count qualifying seasons AND unique teams per player
# ─────────────────────────────────────────────

if qualifying_rows.empty:
    display_df = pd.DataFrame()
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
    
    # 4. Apply your logic: If unique teams > 1, set Team to "2+ Teams"
    display_df["Team"] = display_df.apply(
        lambda x: "2+ Teams" if x["unique_team_count"] > 1 else x["Team"], 
        axis=1
    )
    
    # 5. Sort by season count for the leaderboard
    display_df = display_df.sort_values("season_count", ascending=False).head(MAX_DISPLAY)

# ─────────────────────────────────────────────
#  Build cards
# ─────────────────────────────────────────────

def format_threshold_label(stat, val, op):
    lbl       = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


cards = []
for _, row in display_df.iterrows():
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
        f'width="155" height="155" style="object-fit:cover;border-radius:6px;'
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

# ─────────────────────────────────────────────
#  Title & HTML layout
# ─────────────────────────────────────────────

filter_parts  = [format_threshold_label(s, v, op) for s, op, v in active_filters]
threshold_str = ", ".join(filter_parts)
span_label    = f"{sel_start}–{sel_end}"
pos_val       = st.session_state["hs_position"]
pos_suffix    = f" ({POSITION_OPTIONS[pos_val]})" if pos_val != "all" else ""

page_title = f"Most {threshold_str} Seasons: {span_label}{pos_suffix}"

min_pa_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_pa} PA per season</div>'
    if st.session_state.get("hs_show_min_pa") and min_pa > 0 else ""
)

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
        <p>Data: FanGraphs, Bref</p>
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
.leaderboard-subtitle {{
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
.player-team {{ color: #666; font-size: 0.8rem; margin-bottom: 0.2rem; }}
.season-count {{ font-weight: 800; font-size: 1.05rem; color: #1a1a1a; margin-top: 0.25rem; }}
.season-years {{ color: #aaa; font-size: 0.78rem; margin-top: 0.1rem; line-height: 1.3; }}
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
import re
import html
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

from utils import TEAM_OPTIONS, LEAGUES
from h_utils import (
    get_headshot,
    get_last_updated,
    normalize_team,
    get_team_display,
    format_stat,
    load_final_year as load_hitting_year,
)
from p_utils import load_final_year as load_pitching_year

st.set_page_config(page_title="WAR Leaders", layout="wide", page_icon="⚾")

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
    st.title("WAR Leaders")
with meta_col:
    st.markdown(
        """
        <div class="mobile-meta" style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

current_year = date.today().year
START_YEAR = 1901

last_updated = get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

MODE_SINGLE = "Single Season"
MODE_SPLIT = "Split Seasons"
MODE_MULTI = "Multi-Year Span"

WAR_STATS = ["fWAR", "bWAR", "fWAR-bWAR Avg"]

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
    hit_slim = _slim(load_hitting_year(year), "h")
    pit_slim = _slim(load_pitching_year(year), "p")

    merged = hit_slim.merge(pit_slim, on="PlayerId", how="outer")

    merged["Name"] = merged["Name_h"].combine_first(merged["Name_p"])
    merged["Team"] = merged["Team_h"].combine_first(merged["Team_p"])
    merged["MLBAMID"] = merged["MLBAMID_h"].combine_first(merged["MLBAMID_p"])

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


def load_year_range(start: int, end: int) -> pd.DataFrame:
    frames = []
    for yr in range(start, end + 1):
        df = load_combined_year(yr)
        if df is not None and not df.empty:
            df = df.copy()
            df["Season"] = yr
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_multi_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["fWAR", "bWAR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    last = df.sort_values("Season").groupby("PlayerId", as_index=False).last()[
        ["PlayerId", "Name", "MLBAMID"]
    ]

    team_info = (
        df.groupby("PlayerId")["Team"]
        .apply(lambda teams: (
            "2+ Teams"
            if len({normalize_team(t) for t in teams if pd.notna(t) and str(t).strip() not in ("", "- - -")}) > 1
            else (normalize_team(str(teams.dropna().iloc[0])) if teams.dropna().shape[0] else "N/A")
        ))
        .reset_index()
    )

    summed = df.groupby("PlayerId", as_index=False)[["fWAR", "bWAR"]].sum()

    result = summed.merge(last, on="PlayerId", how="left").merge(team_info, on="PlayerId", how="left")
    result["fWAR-bWAR Avg"] = (result["fWAR"] + result["bWAR"]) / 2
    return result


for key, default in [
    ("wl_view", "Graphic"),
    ("wl_stat", "fWAR"),
    ("wl_mode", MODE_SINGLE),
    ("wl_year", current_year),
    ("wl_start_year", current_year - 1),
    ("wl_end_year", current_year),
    ("wl_team", "all"),
    ("wl_league", "All"),
    ("wl_sort_worst", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2 = st.columns([.5, 2])

with col1:
    view_mode = st.radio(
        "View", ["Graphic", "Database"],
        key="wl_view",
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "Graphic":
        stat = st.selectbox("Stat", WAR_STATS, key="wl_stat")
    else:
        stat = "fWAR"

    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="wl_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, START_YEAR - 1, -1)), key="wl_year")
        sel_start = sel_end = st.session_state["wl_year"]
    else:
        start_options = list(range(current_year, START_YEAR - 1, -1))
        current_start_val = st.session_state.get("wl_start_year", current_year - 1)
        st.selectbox(
            "Start Year", options=start_options,
            index=start_options.index(current_start_val) if current_start_val in start_options else 0,
            key="wl_start_year",
        )
        st.selectbox("End Year", options=list(range(current_year, START_YEAR - 1, -1)), key="wl_end_year")
        sel_start = st.session_state["wl_start_year"]
        sel_end = max(st.session_state["wl_end_year"], sel_start)

    team_disabled = (mode == MODE_MULTI)
    league_disabled = (sel_start < 2013)

    st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key="wl_team",
        disabled=team_disabled,
        help="Team filter unavailable for multi-year span" if team_disabled else None,
    )
    st.selectbox(
        "League",
        options=LEAGUES.keys(),
        key="wl_league",
        disabled=team_disabled or league_disabled,
        help="League filter unavailable for years before 2013 due to possible inaccuracies" if league_disabled else None,
    )

    if view_mode == "Graphic":
        st.checkbox("Show worst", key="wl_sort_worst")

team_val = "all" if team_disabled else st.session_state.get("wl_team", "all")
league_val = "All" if team_disabled or league_disabled else st.session_state.get("wl_league", "All")

# ── Load data ──────────────────────────────────────────────────────────────

if mode == MODE_SINGLE:
    df = load_combined_year(sel_start)
elif mode == MODE_SPLIT:
    df = load_year_range(sel_start, sel_end)
else:
    raw = load_year_range(sel_start, sel_end)
    df = aggregate_multi_year(raw) if raw is not None and not raw.empty else raw

if df is None or df.empty:
    st.error(f"No data found for {sel_start}–{sel_end}.")
    st.stop()

if team_val != "all" and "Team" in df.columns:
    target = normalize_team(team_val)
    df = df[df["Team"].astype(str).apply(lambda t: normalize_team(t) == target)]

if league_val != "All" and "Team" in df.columns:
    league_teams = LEAGUES[league_val]
    df = df[df["Team"].astype(str).apply(lambda t: normalize_team(t) in league_teams)]

if "Team" in df.columns:
    df["TeamDisplay"] = df["Team"].astype(str).apply(get_team_display)
else:
    df["TeamDisplay"] = "N/A"

for s in WAR_STATS:
    df[s] = pd.to_numeric(df[s], errors="coerce")

if df.empty:
    st.error("No players match the selected filters.")
    st.stop()

# ── GRAPHIC VIEW ─────────────────────────────────────────────────────────────

with col2:
    if view_mode == "Graphic":
        sort_worst = st.session_state.get("wl_sort_worst", False)
        ascending = sort_worst  
        df_graphic = df.sort_values(by=stat, ascending=ascending).dropna(subset=[stat]).head(10)

        cards = []
        for _, row in df_graphic.iterrows():
            name = str(row.get("Name", "")).strip()
            team = str(row.get("TeamDisplay", ""))

            if mode == MODE_SPLIT and "Season" in row.index and pd.notna(row.get("Season")):
                team = f"{team} ({int(row['Season'])})"

            raw_val = row.get(stat, np.nan)
            display_val = format_stat(stat, raw_val)
            src = get_headshot(row)

            img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}"/>'
            cards.append(f'''
            <div class="player-card">
              {img_html}
              <div class="player-name">{html.escape(name)}</div>
              <div class="player-team">{html.escape(team)}</div>
              <div class="player-stat">{html.escape(display_val)}</div>
            </div>
            ''')

        span_label = f"{sel_start}" if mode == MODE_SINGLE else f"{sel_start}–{sel_end}"
        team_label = TEAM_OPTIONS.get(team_val, "") if team_val != "all" else ""
        league_label = league_val if league_val != "All" else ""
        mode_label = " Single Season" if mode == MODE_SPLIT else ""
        worst_label = "Worst " if sort_worst else ""

        title = re.sub(
            r"  +", " ",
            f"{span_label}{mode_label} {league_label} {team_label} {worst_label}{stat} Leaders".strip(),
        )

        grid_html = f"""
        <div class="leaderboard-card">
            <div class="leaderboard-title">{html.escape(title)}</div>
            <div class="leaderboard-subtitle">Hitting + Pitching</div>
            <div class="players-grid">{''.join(cards)}</div>
            <div class="footer">
                <p>By: Sox_Savant</p>
                <p></p>
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
html, body {{
    background: transparent;
    font-family: "Source Sans Pro", sans-serif;
    margin: 0;
    padding: 0;
}}
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
    font-size: 0.9rem;
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
.player-name {{ font-weight: 800; margin-top: 0.35rem; font-size: 1rem; }}
.player-team {{ color: #666; font-size: 0.85rem; }}
.player-stat {{ font-weight: 900; font-size: 1.5rem; margin-top: 0.25rem; }}
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
    .leaderboard-card {{ padding: 1rem 0.75rem; border-radius: 10px; }}
    .leaderboard-title {{ font-size: 1.35rem; margin-bottom: 0.6rem; }}
    .leaderboard-subtitle {{ font-size: 0.6rem; margin-top: -0.4rem; }}
    .players-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        grid-auto-rows: auto;
        gap: 0.5rem;
    }}
    .player-card {{ min-width: 130px; flex: 0 0 auto; }}
    .player-card img {{ width: 80px; height: 80px; }}
    .player-name {{ font-size: 0.7rem; }}
    .player-team {{ font-size: 0.7rem; }}
    .player-stat {{ font-size: .9rem; }}
    .footer p    {{ font-size: 0.7rem; }}
    .footer      {{ margin-top: 1rem; }}
}}
</style>
</head>
<body>{grid_html}</body>
</html>
"""
        st.iframe(full_html, height=800)

    # ── DATABASE VIEW ─────────────────────────────────────────────────────────

    else:
        df_db = df.sort_values(by="fWAR", ascending=False, na_position="last").dropna(subset=["fWAR"])

        base_cols = ["Name"]
        if mode == MODE_SPLIT and "Season" in df_db.columns:
            base_cols.append("Season")
        base_cols.append("TeamDisplay")

        display = df_db[[c for c in base_cols if c in df_db.columns] + WAR_STATS].copy()
        display = display.rename(columns={"TeamDisplay": "Team"})
        display = display.reset_index(drop=True)
        display.index += 1

        col_config = {s: st.column_config.NumberColumn(s, format="%.1f") for s in WAR_STATS}

        span_label = f"{sel_start}" if mode == MODE_SINGLE else f"{sel_start}–{sel_end}"
        mode_cap = {
            MODE_SINGLE: "Season",
            MODE_SPLIT: "Split Seasons",
            MODE_MULTI: "Multi-Year Span",
        }[mode]

        st.caption(f"{mode_cap} – {span_label}")
        st.dataframe(display, width="stretch", height=700, column_config=col_config)

        st.markdown(
            "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem;'>"
            "Data: Baseball Reference · FanGraphs · Baseball Savant"
            "</div>",
            unsafe_allow_html=True,
        )
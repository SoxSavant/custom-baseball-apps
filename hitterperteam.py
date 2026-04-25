import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import html
from datetime import date

st.set_page_config(page_title="Hitters Per Team Filter", layout="wide", page_icon="⚾")

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
    st.title("Hitters Per Team Filter")
with meta_col:
    st.markdown(
        '<div style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Imports from h_utils
# ─────────────────────────────────────────────

from h_utils import (
    STAT_ALLOWLIST, SUM_STATS, RATE_STATS, format_stat, STAT_DEFAULTS, MAX_STATS,
    label_map, lower_better, start_year, POSITION_OPTIONS, TEAM_OPTIONS,
    normalize_team, get_team_display, filter_by_position, load_final_year, TEAMS,
)
from utils import get_dynamic_min_pa

# ─────────────────────────────────────────────
#  MLB team ID map for logos
# ─────────────────────────────────────────────

TEAM_MLB_IDS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111,
    "CHC": 112, "CIN": 113, "CLE": 114, "COL": 115,
    "CHW": 145, "DET": 116, "HOU": 117, "KCR": 118,
    "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158,
    "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SDP": 135, "SEA": 136,
    "SFG": 137, "STL": 138, "TBR": 139, "TEX": 140,
    "TOR": 141, "WSN": 120,
}

DIVISION_TEAMS = {
    "AL East":  {"BAL", "BOS", "NYY", "TBR", "TOR"},
    "AL Central": {"CHW", "CLE", "DET", "KCR", "MIN"},
    "AL West":  {"HOU", "LAA", "OAK", "SEA", "TEX"},
    "NL East":  {"ATL", "MIA", "NYM", "PHI", "WSN"},
    "NL Central": {"CHC", "CIN", "MIL", "PIT", "STL"},
    "NL West":  {"ARI", "COL", "LAD", "SDP", "SFG"},
}

ALL_DIVISIONS = list(DIVISION_TEAMS.keys())

def get_team_division(abbrev: str) -> str:
    a = normalize_team(abbrev)
    for div, teams in DIVISION_TEAMS.items():
        if a in teams:
            return div
    return ""

def get_team_logo_url(abbrev: str) -> str:
    mlb_id = TEAM_MLB_IDS.get(normalize_team(abbrev))
    if mlb_id:
        return f"https://www.mlbstatic.com/team-logos/{mlb_id}.svg"
    return ""

# ─────────────────────────────────────────────
#  Constants / modes
# ─────────────────────────────────────────────

MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year
MAX_TEAMS    = 9

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def update_stat_default(i):
    stat = st.session_state[f"tc_stat_{i}"]
    st.session_state[f"tc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"tc_op_{i}"]  = "<=" if stat in lower_better else ">="


def format_threshold(stat: str, val: float, op: str) -> str:
    lbl       = label_map.get(stat, stat)
    formatted = format_stat(stat, val).rstrip("%")
    return f"{formatted}+ {lbl}" if op == ">=" else f"≤ {formatted} {lbl}"


@st.cache_data(show_spinner=False, ttl=3600)
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
    combined = filter_by_position(combined, position)
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


def aggregate_player_group(grp: pd.DataFrame) -> dict:
    result: dict = {}

    if "Name" in grp.columns:
        result["Name"] = str(grp["Name"].dropna().iloc[0]) if not grp["Name"].dropna().empty else ""
    if "PlayerId" in grp.columns:
        ids = grp["PlayerId"].dropna()
        if not ids.empty:
            result["PlayerId"] = ids.iloc[0]
    if "MLBAMID" in grp.columns:
        ids = grp["MLBAMID"].dropna()
        if not ids.empty:
            result["MLBAMID"] = ids.iloc[0]

    if "Team" in grp.columns:
        teams = grp["Team"].dropna().astype(str).tolist()
        result["Team"] = (
            "2+ Teams" if any(get_team_display(t) == "2+ Teams" for t in teams)
            else "2+ Teams" if len({normalize_team(t) for t in teams if t.strip() and t.strip() != "- - -"}) > 1
            else (normalize_team(teams[0]) if teams else "N/A")
        )
    else:
        result["Team"] = "N/A"

    pa_weight = pd.to_numeric(grp["PA"], errors="coerce").fillna(0) if "PA" in grp.columns else pd.Series(np.zeros(len(grp)), index=grp.index)
    pa_total  = pa_weight.sum()

    numeric_cols = [
        col for col in grp.columns
        if pd.api.types.is_numeric_dtype(grp[col]) and col not in {"PlayerId", "MLBAMID", "Season"}
    ]

    for col in numeric_cols:
        series = pd.to_numeric(grp[col], errors="coerce")
        if series.isna().all():
            continue
        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in RATE_STATS and pa_total > 0:
            result[col] = (series * pa_weight).sum(skipna=True) / pa_total
        elif col in MAX_STATS:
            result[col] = series.max(skipna=True)
        else:
            result[col] = (series * pa_weight).sum(skipna=True) / pa_total if pa_total > 0 else series.mean(skipna=True)

    def to_num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    h       = to_num(result.get("H"))
    ab      = to_num(result.get("AB"))
    bb      = to_num(result.get("BB"))
    hbp     = to_num(result.get("HBP"))
    sf      = to_num(result.get("SF"))
    doubles = to_num(result.get("2B"))
    triples = to_num(result.get("3B"))
    hr      = to_num(result.get("HR"))

    if pd.notna(h) and pd.notna(doubles) and pd.notna(triples) and pd.notna(hr):
        singles = h - doubles - triples - hr
        result["1B"]  = singles if singles >= 0 else np.nan
        result["XBH"] = doubles + triples + hr
        tb = singles + 2 * doubles + 3 * triples + 4 * hr
        result["TB"] = tb
        if pd.notna(ab) and ab > 0:
            result["AVG"] = h / ab
            result["SLG"] = tb / ab

    bb_v    = 0 if pd.isna(bb)  else bb
    hbp_v   = 0 if pd.isna(hbp) else hbp
    sf_v    = 0 if pd.isna(sf)  else sf
    obp_den = (ab if pd.notna(ab) else 0) + bb_v + hbp_v + sf_v
    if obp_den > 0 and pd.notna(h):
        result["OBP"] = (h + bb_v + hbp_v) / obp_den

    slg = result.get("SLG")
    obp = result.get("OBP")
    avg = result.get("AVG")
    if pd.notna(slg) and pd.notna(obp):
        result["OPS"] = slg + obp
    if pd.notna(slg) and pd.notna(avg):
        result["ISO"] = slg - avg

    return result

# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

for key, default in [
    ("tc_year",         current_year),
    ("tc_start_year",   current_year - 1),
    ("tc_end_year",     current_year),
    ("tc_mode",         MODE_SINGLE),
    ("tc_position",     "all"),
    ("tc_show_min_pa",  True),
    ("tc_show_player_pa", False),
    ("tc_show_all_teams", False),
    ("tc_val_0",        .350),
    ("tc_val_1",        125),
]:
    if key not in st.session_state:
        st.session_state[key] = default

for div in ALL_DIVISIONS:
    key = f"tc_div_{div}"
    if key not in st.session_state:
        st.session_state[key] = True

# ─────────────────────────────────────────────
#  Sidebar controls
# ─────────────────────────────────────────────

col1, col2 = st.columns([0.5, 2])

with col1:
    num_stats = st.radio(
        "Number of stat filters", [1, 2, 3, 4],
        index=1, horizontal=True, key="tc_num_stats",
    )

    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="tc_mode")

    if mode == MODE_MULTI:
        st.caption("Multi-year excludes seasons ofplayers who switched teams mid-season in any year of the span.")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year - 1, -1)), key="tc_year")
        s_year = st.session_state["tc_year"]
        e_year = st.session_state["tc_year"]

        if "tc_last_year" not in st.session_state:
            st.session_state.tc_last_year = s_year
        if s_year != st.session_state.tc_last_year:
            st.session_state["tc_min_pa"] = get_dynamic_min_pa(s_year)
            st.session_state.tc_last_year = s_year
    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key="tc_start_year")
        st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key="tc_end_year")
        s_year = st.session_state["tc_start_year"]
        e_year = max(st.session_state["tc_end_year"], s_year)

    if "tc_min_pa" not in st.session_state:
        st.session_state["tc_min_pa"] = get_dynamic_min_pa(
            s_year if mode == MODE_SINGLE else current_year
        )

    st.number_input("Min PA", min_value=0, max_value=20000, key="tc_min_pa")

    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        default_stat  = "xwOBA" if i == 0 else "wRC+" if i == 1 else STAT_ALLOWLIST[0]
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
            st.selectbox("Op", [">=", "<="], key=f"tc_op_{i}", index=0, label_visibility="collapsed")
        with val_col:
            RATE_STATS_3DP = {"AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "ISO", "BABIP"}
            if new_stat in RATE_STATS_3DP or new_stat == "wOBA-xwOBA":
                step, fmt = 0.001, "%.3f"
            elif "%" in new_stat or new_stat in {"EV", "fWAR", "bWAR", "BatSpd", "Def", "Off"}:
                step, fmt = 0.1, "%.1f"
            elif new_stat in {"WPA", "Clutch"}:
                step, fmt = 0.01, "%.2f"
            else:
                step, fmt = 1.0, "%.0f"
            st.number_input(
                f"Value {i+1}", step=step, key=f"tc_val_{i}",
                label_visibility="collapsed", format=fmt,
            )

    st.selectbox(
        "Position", options=list(POSITION_OPTIONS.keys()),
        format_func=lambda x: POSITION_OPTIONS[x], key="tc_position",
    )

    st.checkbox("Show min PA",      key="tc_show_min_pa")
    st.checkbox("Show player PA",   key="tc_show_player_pa")
    st.checkbox("Show all 30 teams", key="tc_show_all_teams")

    st.markdown("**Divisions**")
    for div in ALL_DIVISIONS:
        st.checkbox(div, key=f"tc_div_{div}")

# ─────────────────────────────────────────────
#  Load & filter data
# ─────────────────────────────────────────────

min_pa_val   = int(st.session_state["tc_min_pa"])
position_val = st.session_state["tc_position"]

df = load_data(s_year, e_year, mode, position=position_val)

if df is None or df.empty:
    st.error(f"No data found for {s_year}–{e_year}.")
    st.stop()

# Min PA
if min_pa_val > 0 and "PA" in df.columns:
    df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa_val]

# Position filter — skip for MULTI (already applied inside load_data)
if mode != MODE_MULTI:
    df = filter_by_position(df, position_val)

# Team display column
if "Team" in df.columns:
    df["TeamDisplay"] = df["Team"].astype(str).apply(get_team_display)
else:
    df["TeamDisplay"] = "N/A"

# ─────────────────────────────────────────────
#  Active filters
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
#  Group by team (and year for split mode)
# ─────────────────────────────────────────────

show_all_teams   = st.session_state.get("tc_show_all_teams", False)
active_divisions = [div for div in ALL_DIVISIONS if st.session_state.get(f"tc_div_{div}", True)]

sort_stat = active_filters[0][0] if active_filters else None
sort_op   = active_filters[0][1] if active_filters else ">="
sort_asc_tiebreak = sort_stat in lower_better and sort_op == "<=" if sort_stat else False

team_groups = []

if mode == MODE_SPLIT:
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
    if not show_all_teams:
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
    if not show_all_teams:
        team_groups = team_groups[:MAX_TEAMS]

# ─────────────────────────────────────────────
#  Build HTML
# ─────────────────────────────────────────────

filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str   = ", ".join(filter_parts)
span_label   = f"{s_year}" if mode == MODE_SINGLE else f"{s_year}–{e_year}"
mode_label   = " (Single Season)" if mode == MODE_SPLIT else ""
pos_suffix   = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
middle_label = " in " if mode == MODE_SINGLE else ": "
title_text   = f"Most Hitters with {filter_str}{middle_label}{span_label}{mode_label}{pos_suffix}"

min_pa_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_pa_val} PA</div>'
    if st.session_state.get("tc_show_min_pa") else ""
)

show_player_pa = st.session_state.get("tc_show_player_pa", False)

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
            f'width="38" height="38" style="object-fit:contain;display:block;"/>'
            if logo_url else
            f'<div style="width:38px;height:38px;display:flex;align-items:center;'
            f'justify-content:center;font-size:10px;font-weight:700;color:#888;">'
            f'{html.escape(abbrev)}</div>'
        )

        # Player rows — name on left, stats on right
        player_rows_html = ""
        for _, prow in tg["players"].iterrows():
            pname = html.escape(str(prow.get("Name", "")).strip())
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
            if show_player_pa:
                pa_val = prow.get("PA", np.nan)
                if pd.notna(pa_val):
                    pa_html = f'<span class="p-pa">{int(pa_val)}</span>'

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
    {min_pa_subtitle}
    <div class="teams-grid">{team_card_html}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p>Data: FanGraphs, Bref</p>
    </div>
</div>
"""

# Height: title area + rows of cards (4 per row), each card height depends on player count
cards_per_row   = 3
num_rows        = max(1, -(-num_teams_shown // cards_per_row))  # ceiling division
avg_players     = total_players_shown / max(num_teams_shown, 1)
card_height_est = 90 + avg_players * 26
est_height      = 220 + int(num_rows * card_height_est) + 80

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
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
    margin-top: 1.5rem;
    padding: 0 0.5rem;
}}
.footer p {{
    margin: 0;
    font-size: .95rem;
    color: #6d7075;
    flex: 1;
    text-align: center;
}}
.footer p:first-child {{ text-align: left; }}
.footer p:last-child  {{ text-align: right; }}
</style>
</head>
<body>{grid_html}</body>
</html>"""

with col2:
    components.html(full_html, height=est_height, scrolling=True)
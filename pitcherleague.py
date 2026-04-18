import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import unicodedata
import html
import re
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Pitcher Stat League Leaders", layout="wide", page_icon="⚾")

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
    st.title("Pitcher Stat League Leaders")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Imports from shared utils
# ─────────────────────────────────────────────

from p_utils import (
    STAT_ALLOWLIST, SUM_STATS, RATE_STATS, start_year,
    get_headshot, label_map, lower_better,
    TEAM_OPTIONS, normalize_team, get_team_display,
    outs_to_ip, ip_to_outs, format_stat,
)
from utils import get_dynamic_min_ip

# ─────────────────────────────────────────────
#  Stat presets
# ─────────────────────────────────────────────

PRESETS = {
    "Statcast": [
        "xERA", "xFIP", "vFA", "EV", "Chase%", "Whiff%", "K%","BB%","Barrel%", "HardHit%", 
    ],
    "Standard": ["fWAR","bWAR", "ERA","ERA-","FIP","FIP-", "IP","SO","BB","WHIP"],
    "More Stats": ["xERA", "xFIP","SO","fWAR","BB/9","HR/9","K/BB","SIERA"],
    "Empty – Add your own": [],
}

MAX_DISPLAY_STATS = 10   # grid cap (5 × 2)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year

# ─────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_final_year(year: int) -> pd.DataFrame:
    path = f"data/final/pitching_final_{year}.csv"
    try:
        df = pd.read_csv(path)
        df["Season"] = year
        return df
    except Exception:
        return pd.DataFrame()


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
        result["Team"] = "2+ Teams" if any(get_team_display(t) == "2+ Teams" for t in teams) else (
            "2+ Teams" if len({normalize_team(t) for t in teams if t.strip() and t.strip() != "- - -"}) > 1
            else (normalize_team(teams[0]) if teams else "N/A")
        )
    else:
        result["Team"] = "N/A"

    # IP: sum outs then convert back
    ip_outs_total = np.nan
    if "IP" in grp.columns:
        outs = pd.to_numeric(grp["IP"], errors="coerce").apply(ip_to_outs).dropna()
        if not outs.empty:
            ip_outs_total = outs.sum()
            result["IP"] = outs_to_ip(ip_outs_total)

    # Weight by IP outs for rate stats
    if "IP" in grp.columns:
        weight = pd.to_numeric(grp["IP"], errors="coerce").apply(ip_to_outs).fillna(0)
    elif "TBF" in grp.columns:
        weight = pd.to_numeric(grp["TBF"], errors="coerce").fillna(0)
    else:
        weight = pd.Series(np.ones(len(grp)), index=grp.index)
    weight_total = weight.sum()

    numeric_cols = [
        col for col in grp.columns
        if pd.api.types.is_numeric_dtype(grp[col])
        and col not in {"PlayerId", "MLBAMID", "Season", "IP"}
    ]

    total_er = np.nan
    for col in numeric_cols:
        series = pd.to_numeric(grp[col], errors="coerce")
        if series.isna().all():
            continue
        if col == "ER":
            total_er = series.sum(skipna=True)
        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in RATE_STATS and weight_total > 0:
            result[col] = (series * weight).sum(skipna=True) / weight_total
        else:
            result[col] = series.mean(skipna=True)

    # Recompute ERA from totals
    if pd.notna(total_er) and not pd.isna(ip_outs_total) and ip_outs_total > 0:
        result["ERA"] = (total_er / (ip_outs_total / 3)) * 9

    return result


@st.cache_data(show_spinner=False, ttl=3600)
def load_data(s_year: int, e_year: int, mode: str) -> pd.DataFrame:
    if mode == MODE_SINGLE:
        return load_final_year(s_year)

    frames = []
    for year in range(s_year, e_year + 1):
        df = load_final_year(year)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if mode == MODE_SPLIT:
        return combined

    if "PlayerId" not in combined.columns:
        return combined

    grouped_rows = []
    for _, grp in combined.groupby("PlayerId"):
        grouped_rows.append(aggregate_player_group(grp))

    return pd.DataFrame(grouped_rows)


# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

min_ip = get_dynamic_min_ip(current_year)
default_stats = list(PRESETS["Statcast"])

for key, default in [
    ("ll_year",        current_year),
    ("ll_start_year",  current_year - 1),
    ("ll_end_year",    current_year),
    ("ll_min_ip",      min_ip),
    ("ll_team",        "all"),
    ("ll_mode",        MODE_SINGLE),
    ("ll_show_worst",  False),
    ("ll_show_min_ip", True),
    ("ll_stats",       default_stats),
    ("ll_preset",      "Statcast"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Stat builder callbacks
# ─────────────────────────────────────────────

SENTINEL_ADD    = "Add"
SENTINEL_REMOVE = "Remove"
ADD_KEY         = "ll_add_stat_select"
REMOVE_KEY      = "ll_remove_stat_select"
ADD_RESET_KEY   = "ll_add_reset"
REMOVE_RESET_KEY = "ll_remove_reset"


def _add_stat_cb():
    choice = st.session_state.get(ADD_KEY)
    if not choice or choice == SENTINEL_ADD:
        return
    current = list(st.session_state["ll_stats"])
    if choice not in current and len(current) < MAX_DISPLAY_STATS:
        current.append(choice)
        st.session_state["ll_stats"] = current
    st.session_state[ADD_RESET_KEY] = True


def _remove_stat_cb():
    choice = st.session_state.get(REMOVE_KEY)
    if not choice or choice == SENTINEL_REMOVE:
        return
    current = list(st.session_state["ll_stats"])
    if choice in current:
        current.remove(choice)
    st.session_state["ll_stats"] = current
    st.session_state[REMOVE_RESET_KEY] = True


def _move_stat(idx: int, delta: int):
    stats = list(st.session_state["ll_stats"])
    target = idx + delta
    if 0 <= target < len(stats):
        stats[idx], stats[target] = stats[target], stats[idx]
        st.session_state["ll_stats"] = stats


def _apply_preset_cb():
    preset_name = st.session_state.get("ll_preset_select")
    if not preset_name or preset_name not in PRESETS:
        return
    valid = [s for s in PRESETS[preset_name] if s in STAT_ALLOWLIST]
    st.session_state["ll_stats"]  = valid[:MAX_DISPLAY_STATS]
    st.session_state["ll_preset"] = preset_name
    st.session_state[ADD_RESET_KEY]    = True
    st.session_state[REMOVE_RESET_KEY] = True


# ─────────────────────────────────────────────
#  Controls
# ─────────────────────────────────────────────

col1, col2 = st.columns([0.5, 2])

with col1:
    # ── Preset selector ──
    st.markdown("**Presets**")
    preset_options = list(PRESETS.keys())
    prior_preset   = st.session_state.get("ll_preset", preset_options[0])
    preset_index   = preset_options.index(prior_preset) if prior_preset in preset_options else 0

    st.selectbox(
        "Choose a stat view:",
        preset_options,
        index=preset_index,
        key="ll_preset_select",
        on_change=_apply_preset_cb,
        label_visibility="collapsed",
    )

    # ── Mode / year ──
    mode = st.radio("Mode", options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI], key="ll_mode")

    if mode == MODE_SINGLE:
        st.selectbox("Year", options=list(range(current_year, start_year - 1, -1)), key="ll_year")
        s_year = st.session_state["ll_year"]
        e_year = st.session_state["ll_year"]

        if "ll_last_year" not in st.session_state:
            st.session_state.ll_last_year = s_year
        if s_year != st.session_state.ll_last_year:
            st.session_state["ll_min_ip"] = get_dynamic_min_ip(s_year)
            st.session_state.ll_last_year = s_year
    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key="ll_start_year")
        st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key="ll_end_year")
        s_year = st.session_state["ll_start_year"]
        e_year = max(st.session_state["ll_end_year"], s_year)

    st.number_input("Min IP", min_value=0, max_value=5000, key="ll_min_ip")

    team_disabled = (mode == MODE_MULTI)
    st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key="ll_team",
        disabled=team_disabled,
        help="Team filter unavailable for multi-year span" if team_disabled else None,
    )

    st.checkbox("Show worst",  key="ll_show_worst")
    st.checkbox("Show min IP", key="ll_show_min_ip")

    # ── Dynamic stat builder ──
    st.markdown("---")
    st.markdown("**Stats** (up to 10)")

    current_stats = list(st.session_state["ll_stats"])

    in_list      = current_stats
    not_in_list  = [s for s in STAT_ALLOWLIST if s not in in_list]
    add_options  = [SENTINEL_ADD]    + not_in_list
    rem_options  = [SENTINEL_REMOVE] + in_list

    if st.session_state.pop(ADD_RESET_KEY, False):
        st.session_state[ADD_KEY] = SENTINEL_ADD
    if st.session_state.pop(REMOVE_RESET_KEY, False):
        st.session_state[REMOVE_KEY] = SENTINEL_REMOVE

    if st.session_state.get(ADD_KEY) not in add_options:
        st.session_state[ADD_KEY] = SENTINEL_ADD
    if st.session_state.get(REMOVE_KEY) not in rem_options:
        st.session_state[REMOVE_KEY] = SENTINEL_REMOVE

    add_col, rem_col = st.columns(2)
    with add_col:
        st.selectbox(
            "Add",
            add_options,
            key=ADD_KEY,
            format_func=lambda x: label_map.get(x, x),
            on_change=_add_stat_cb,
            disabled=(len(current_stats) >= MAX_DISPLAY_STATS),
        )
    with rem_col:
        st.selectbox(
            "Remove",
            rem_options,
            key=REMOVE_KEY,
            format_func=lambda x: label_map.get(x, x),
            on_change=_remove_stat_cb,
        )

    current_stats = list(st.session_state["ll_stats"])

    hdr_a, hdr_b, hdr_c = st.columns([0.3, 0.3, 1])
    hdr_a.markdown("**▲**")
    hdr_b.markdown("**▼**")
    hdr_c.markdown("**Stat**")

    for i, stat in enumerate(current_stats):
        up_col, dn_col, nm_col = st.columns([0.3, 0.3, 1])
        with up_col:
            st.button("▲", key=f"ll_up_{i}", disabled=(i == 0),
                      on_click=_move_stat, args=(i, -1))
        with dn_col:
            st.button("▼", key=f"ll_dn_{i}", disabled=(i == len(current_stats) - 1),
                      on_click=_move_stat, args=(i, 1))
        nm_col.write(label_map.get(stat, stat))

    selected_stats = list(st.session_state["ll_stats"])


# ─────────────────────────────────────────────
#  Resolve filter values
# ─────────────────────────────────────────────

min_ip_val   = int(st.session_state.get("ll_min_ip", 0))
team_val     = "all" if team_disabled else st.session_state.get("ll_team", "all")
show_worst   = st.session_state.get("ll_show_worst", False)

# ─────────────────────────────────────────────
#  Load & filter data
# ─────────────────────────────────────────────

df = load_data(s_year, e_year, mode)
if df is None or df.empty:
    st.error(f"No data found for {s_year}–{e_year}.")
    st.stop()

if min_ip_val > 0 and "IP" in df.columns:
    df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip_val]

if team_val != "all" and "Team" in df.columns:
    target = normalize_team(team_val)
    df = df[df["Team"].astype(str).apply(lambda t: normalize_team(t) == target)]

if "Team" in df.columns:
    df["TeamDisplay"] = df["Team"].astype(str).apply(get_team_display)
else:
    df["TeamDisplay"] = "N/A"

# ─────────────────────────────────────────────
#  Build one leader per stat
# ─────────────────────────────────────────────

def get_leader(df: pd.DataFrame, stat: str, show_worst: bool):
    if stat not in df.columns:
        return None
    col = pd.to_numeric(df[stat], errors="coerce")
    valid = df[col.notna()].copy()
    valid[stat] = col[col.notna()]
    if valid.empty:
        return None
    stat_lower = stat in lower_better
    ascending = (stat_lower and not show_worst) or (not stat_lower and show_worst)
    return valid.sort_values(by=stat, ascending=ascending).iloc[0]


leader_rows = []
for stat in selected_stats:
    row = get_leader(df, stat, show_worst)
    leader_rows.append((stat, row))

num_stats = len(selected_stats)

# ─────────────────────────────────────────────
#  Build HTML cards — dynamic grid columns
# ─────────────────────────────────────────────

if num_stats <= 5:
    grid_cols = num_stats
elif num_stats <= 10:
    grid_cols = (num_stats + 1) // 2
else:
    grid_cols = 5


def make_card(stat, row):
    stat_label = html.escape(label_map.get(stat, stat))
    if row is None:
        return f'''
        <div class="player-card">
          <div class="card-no-data">No data</div>
          <div class="card-stat-line">{stat_label}: —</div>
        </div>
        '''

    name = str(row.get("Name", "")).strip()
    team = str(row.get("TeamDisplay", ""))
    raw_val = row.get(stat, np.nan)
    display_val = format_stat(stat, raw_val)
    src = get_headshot(row)

    year_html = ""
    if mode == MODE_SPLIT and "Season" in row.index and pd.notna(row.get("Season")):
        year_html = f'<div class="player-season">({int(row["Season"])})</div>'

    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}"/>'
    return f'''
    <div class="player-card">
      {img_html}
      <div class="player-stat-line">{stat_label}: <span class="stat-val">{html.escape(display_val)}</span></div>
      <div class="player-name-team">{html.escape(name)} | {html.escape(team)}</div>
      {year_html}
    </div>
    '''


cards = [make_card(s, r) for s, r in leader_rows]

# ─────────────────────────────────────────────
#  Title
# ─────────────────────────────────────────────

span_label  = f"{s_year}" if mode == MODE_SINGLE else f"{s_year}–{e_year}"
team_label  = f"{TEAM_OPTIONS.get(team_val, '')}" if team_val != "all" else ""
mode_label  = " (Single Season) " if mode == MODE_SPLIT else ""
worst_label = "Worst" if show_worst else "Best"
middle_label = " in " if mode == MODE_SINGLE else ": "

title = re.sub(
    r"  +", " ",
    f"{worst_label} {team_label} Pitchers{middle_label}{span_label}{mode_label}".strip()
)

min_ip_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_ip_val} IP</div>'
    if st.session_state.get("ll_show_min_ip") else ""
)

# ─────────────────────────────────────────────
#  Render
# ─────────────────────────────────────────────

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_ip_subtitle}
    <div class="players-grid" style="grid-template-columns: repeat({grid_cols}, minmax(0, 1fr));">{''.join(cards)}</div>
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
html, body {{ background: transparent; font-family: "Source Sans Pro", sans-serif; margin: 0; padding: 0; }}
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
.player-name-team {{
    font-weight: 500;
    font-size: 0.8rem;
    margin-top: 0.35rem;
    color: #222;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 155px;
}}
.player-season {{
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.1rem;
}}
.player-stat-line {{
    font-weight: 700;
    font-size: 1.05rem;
    color: #444;
    margin-top: 0.1rem;
}}
.stat-val {{
    font-weight: 900;
    font-size: 1.2rem;
    color: #000;
}}
.card-no-data {{ color: #aaa; font-size: 0.9rem; margin-top: 0.5rem; }}
.card-stat-line {{ font-size: 0.95rem; color: #999; margin-top: 0.15rem; }}
.footer {{ display: flex; justify-content: space-between; align-items: center; margin-top: .5rem; }}
.footer p {{ margin: 0; font-size: 0.9rem; color: #666; flex: 1; text-align: center; }}
.footer p:first-child {{ text-align: left; }}
.footer p:last-child {{ text-align: right; }}
</style>
</head>
<body>{grid_html}</body>
</html>
"""

with col2:
    if grid_cols > 0:
        num_rows = (num_stats + grid_cols - 1) // grid_cols
    else:
        num_rows = 1
    card_height = 220 * num_rows + 400
    components.html(full_html, height=card_height)
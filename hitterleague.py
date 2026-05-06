import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import unicodedata
import html
import re
from datetime import date

st.set_page_config(page_title="Hitter League Leaders", layout="wide", page_icon="⚾")

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
    st.title("Hitter League Leaders")
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

from h_utils import (
    STAT_ALLOWLIST, SUM_STATS, RATE_STATS, format_stat, start_year, MAX_STATS,
    get_headshot, label_map, lower_better,
    POSITION_OPTIONS, TEAM_OPTIONS, normalize_team, get_team_display, filter_by_position, load_final_year, aggregate_player_group
)
from utils import get_dynamic_min_pa

# ─────────────────────────────────────────────
#  Stat presets  (no longer capped at 10)
# ─────────────────────────────────────────────

PRESETS = {
    "Statcast": [
        "xwOBA", "xBA","xSLG","EV", "Barrel%", "HardHit%", "BatSpd", "Squared-Up%", "Chase%", "Whiff%",
    ],
    "Standard": ["AVG", "OBP", "SLG", "OPS", "HR", "RBI", "R", "SB", "2B", "3B"],
    "Value":    ["fWAR", "bWAR", "Off", "Def", "WPA", "Clutch"],
    "Defense": ["DRS","OAA","FRV","FRM","Def"],
    "Discipline": ["Chase%","Swing%", "Z-Swing%", "O-Contact%","Z-Contact%","Whiff%","Zone%","K%","BB%","BB/K"],
    "Counting Stats": ["G", "PA", "AB", "H","R", "RBI", "HR", "XBH", "TB", "SB"],
    "Offensive Stats": ["wRC+", "wOBA", "xwOBA", "wOBA-xwOBA",  "AVG", "OBP", "SLG","ISO", "OPS", 
    "BABIP",],
    "Empty – Add your own": []
}

MAX_DISPLAY_STATS = 10   # grid cap (5 × 2)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

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



def load_data(s_year: int, e_year: int, mode: str, position: str = "all") -> pd.DataFrame:
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

    combined = filter_by_position(combined, position)
    if combined.empty:
        return pd.DataFrame()

    grouped_rows = []
    for _, grp in combined.groupby("PlayerId"):
        grouped_rows.append(aggregate_player_group(grp))

    return pd.DataFrame(grouped_rows)


# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

min_pa = get_dynamic_min_pa(current_year)
default_stats = list(PRESETS["Statcast"])

for key, default in [
    ("ll_year",        current_year),
    ("ll_start_year",  current_year - 1),
    ("ll_end_year",    current_year),
    ("ll_min_pa",      min_pa),
    ("ll_position",    "all"),
    ("ll_team",        "all"),
    ("ll_mode",        MODE_SINGLE),
    ("ll_show_worst",  False),
    ("ll_show_min_pa", True),
    ("ll_stats",       default_stats),
    ("ll_preset",      "Statcast"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Stat builder callbacks  (mirroring comp app)
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
            st.session_state["ll_min_pa"] = get_dynamic_min_pa(s_year)
            st.session_state.ll_last_year = s_year
    else:
        st.selectbox("Start Year", options=list(range(current_year, start_year - 1, -1)), key="ll_start_year")
        st.selectbox("End Year",   options=list(range(current_year, start_year - 1, -1)), key="ll_end_year")
        s_year = st.session_state["ll_start_year"]
        e_year = max(st.session_state["ll_end_year"], s_year)

    st.number_input("Min PA", min_value=0, max_value=20000, key="ll_min_pa")

    st.selectbox(
        "Position",
        options=list(POSITION_OPTIONS.keys()),
        format_func=lambda x: POSITION_OPTIONS[x],
        key="ll_position",
    )

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
    st.checkbox("Show min PA", key="ll_show_min_pa")

    # ── Dynamic stat builder ──
    st.markdown("---")
    st.markdown("**Stats** (up to 10)")

    current_stats = list(st.session_state["ll_stats"])

    # Add / Remove dropdowns
    in_list      = current_stats
    not_in_list  = [s for s in STAT_ALLOWLIST if s not in in_list]
    add_options  = [SENTINEL_ADD]    + not_in_list
    rem_options  = [SENTINEL_REMOVE] + in_list

    # Reset sentinels after add/remove
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

    # Re-read after possible add/remove
    current_stats = list(st.session_state["ll_stats"])

    # Reorder rows
    hdr_a, hdr_b, hdr_c = st.columns([0.3, 0.3, 1])
    hdr_a.markdown("**▲**")
    hdr_b.markdown("**▼**")
    hdr_c.markdown("**Stat**")

    for i, stat in enumerate(current_stats):
        up_col, dn_col, nm_col = st.columns([0.3, 0.3, 1])
        with up_col:
            st.button("▲", key=f"ll_up_{i}",   disabled=(i == 0),
                      on_click=_move_stat, args=(i, -1))
        with dn_col:
            st.button("▼", key=f"ll_dn_{i}",   disabled=(i == len(current_stats) - 1),
                      on_click=_move_stat, args=(i,  1))
        nm_col.write(label_map.get(stat, stat))

    selected_stats = list(st.session_state["ll_stats"])


# ─────────────────────────────────────────────
#  Resolve filter values
# ─────────────────────────────────────────────

min_pa_val   = int(st.session_state.get("ll_min_pa", 0))
position_val = st.session_state.get("ll_position", "all")
team_val     = "all" if team_disabled else st.session_state.get("ll_team", "all")
show_worst   = st.session_state.get("ll_show_worst", False)

# ─────────────────────────────────────────────
#  Load & filter data
# ─────────────────────────────────────────────

df = load_data(s_year, e_year, mode, position_val)
if df is None or df.empty:
    st.error(f"No data found for {s_year}–{e_year}.")
    st.stop()

if min_pa_val > 0 and "PA" in df.columns:
    df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa_val]

if mode != MODE_MULTI:
    df = filter_by_position(df, position_val)

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
#  Build HTML cards  — dynamic grid columns
# ─────────────────────────────────────────────

# Choose columns: ≤5 → single row; 6-10 → two rows of ceil(n/2)
if num_stats <= 5:
    grid_cols = num_stats
elif num_stats <= 10:
    grid_cols = (num_stats + 1) // 2  # ceil(n/2), so rows balance nicely
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
pos_suffix  = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
team_label  = f"{TEAM_OPTIONS.get(team_val, "")}" if team_val != "all" else ""
mode_label  = " (Single Season) " if mode == MODE_SPLIT else ""
worst_label = "(Worst)" if show_worst else ""


overall_label = "Stat Leaders" 

title = re.sub(
    r"  +", " ",
    f"{team_label}{span_label} {overall_label} {mode_label}{worst_label} {pos_suffix}".strip()
)

min_pa_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_pa_val} PA</div>'
    if st.session_state.get("ll_show_min_pa") else ""
)

# ─────────────────────────────────────────────
#  Render
# ─────────────────────────────────────────────

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_pa_subtitle}
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
.player-season {{
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.1rem;
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
    # Height scales with number of rows
    if grid_cols > 0:
        num_rows = (num_stats + grid_cols - 1) // grid_cols
    else:
        num_rows = 1
    card_height = 220 * num_rows + 400
    components.html(full_html, height=card_height)
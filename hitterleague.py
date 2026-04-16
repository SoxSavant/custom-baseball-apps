import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import unicodedata
import html
import re
from pathlib import Path
from datetime import date

st.set_page_config(page_title="League Leaders", layout="wide", page_icon="⚾")

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
    st.title("League Leaders")
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
    POSITION_OPTIONS, TEAM_OPTIONS, normalize_team, get_team_display, filter_by_position
)
from utils import get_dynamic_min_pa

# ─────────────────────────────────────────────
#  Stat presets
# ─────────────────────────────────────────────

PRESETS = {
    "Statcast": [
         "xwOBA","EV",  "Barrel%", "HardHit%","BatSpd","Squared-Up%",  "Chase%", "Whiff%","K%", "BB%"
    ],
    "Standard":["AVG","OBP","SLG","OPS","HR","RBI","R","SB","2B","3B"],
    "Value":["fWAR","bWAR","Off","Def","BsR","WPA","Clutch"]
   
}

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year
NUM_STATS = 10  # how many stats shown as league leaders

# ─────────────────────────────────────────────
#  Data loading (identical to original app)
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


def normalize_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\xa0", " ").strip()
    try:
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(cleaned.split()).lower()



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

    pa_weight = pd.to_numeric(grp["PA"], errors="coerce").fillna(0) if "PA" in grp.columns else pd.Series(np.zeros(len(grp)), index=grp.index)
    pa_total = pa_weight.sum()

    numeric_cols = [
        col for col in grp.columns
        if pd.api.types.is_numeric_dtype(grp[col])
        and col not in {"PlayerId", "MLBAMID", "Season"}
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

    h = to_num(result.get("H"))
    ab = to_num(result.get("AB"))
    bb = to_num(result.get("BB"))
    hbp = to_num(result.get("HBP"))
    sf = to_num(result.get("SF"))
    doubles = to_num(result.get("2B"))
    triples = to_num(result.get("3B"))
    hr = to_num(result.get("HR"))

    if pd.notna(h) and pd.notna(doubles) and pd.notna(triples) and pd.notna(hr):
        singles = h - doubles - triples - hr
        result["1B"] = singles if singles >= 0 else np.nan
        result["XBH"] = doubles + triples + hr
        tb = singles + 2 * doubles + 3 * triples + 4 * hr
        result["TB"] = tb
        if pd.notna(ab) and ab > 0:
            result["AVG"] = h / ab
            result["SLG"] = tb / ab

    bb_v = 0 if pd.isna(bb) else bb
    hbp_v = 0 if pd.isna(hbp) else hbp
    sf_v = 0 if pd.isna(sf) else sf
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


@st.cache_data(show_spinner=False, ttl=3600)
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
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Controls
# ─────────────────────────────────────────────

col1, col2 = st.columns([0.5, 2])

with col1:
    # ── Preset buttons ──
    st.markdown("**Presets**")
    options =  list(PRESETS.keys())

    selected_preset = st.selectbox(
    "Choose a stat view:", 
    options,
    label_visibility="collapsed" 
)

    preset_stats = PRESETS[selected_preset]
    
    valid = [s for s in preset_stats if s in STAT_ALLOWLIST]
    
    if st.session_state.get("ll_stats") != valid[:NUM_STATS]:
        st.session_state["ll_stats"] = valid[:NUM_STATS]
        st.rerun()


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

    st.markdown("**Stats (pick 10)**")
    current_stats = st.session_state["ll_stats"]

    new_stats = []
    for i in range(NUM_STATS):
        current_val = current_stats[i] if i < len(current_stats) else STAT_ALLOWLIST[0]
        chosen = st.selectbox(
            f"Stat {i + 1}",
            options=STAT_ALLOWLIST,
            index=STAT_ALLOWLIST.index(current_val) if current_val in STAT_ALLOWLIST else 0,
            format_func=lambda x: label_map.get(x, x),
            key=f"ll_stat_{i}",
        )
        new_stats.append(chosen)

    st.session_state["ll_stats"] = new_stats


# ─────────────────────────────────────────────
#  Resolve filter values
# ─────────────────────────────────────────────

min_pa_val   = int(st.session_state.get("ll_min_pa", 0))
position_val = st.session_state.get("ll_position", "all")
team_val     = "all" if team_disabled else st.session_state.get("ll_team", "all")
show_worst   = st.session_state.get("ll_show_worst", False)
selected_stats = st.session_state["ll_stats"]

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


# ─────────────────────────────────────────────
#  Build HTML cards  (5×2 headshot grid)
# ─────────────────────────────────────────────

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

    if mode == MODE_SPLIT and "Season" in row.index and pd.notna(row.get("Season")):
        team = f"{team} ({int(row['Season'])})"

    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}"/>'
    return f'''
    <div class="player-card">
      {img_html}
      <div class="player-stat-line">{stat_label}: <span class="stat-val">{html.escape(display_val)}</span></div>
      <div class="player-name-team">{html.escape(name)} | {html.escape(team)}</div>
    </div>
    '''


cards = [make_card(s, r) for s, r in leader_rows]

# ─────────────────────────────────────────────
#  Title
# ─────────────────────────────────────────────

span_label  = f"{s_year}" if mode == MODE_SINGLE else f"{s_year}–{e_year}"
pos_suffix  = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
team_label  = TEAM_OPTIONS.get(team_val, "") if team_val != "all" else ""
mode_label  = " Single Season" if mode == MODE_SPLIT else ""
worst_label = "Worst " if show_worst else ""

title = re.sub(
    r"  +", " ",
    f"{span_label}{mode_label} {team_label} {worst_label}Stat Leaders{pos_suffix}".strip()
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
    components.html(full_html, height=800)
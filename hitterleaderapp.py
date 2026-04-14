import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import unicodedata
import html
import re
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Custom Hitting Leaderboard", layout="wide", page_icon="⚾")

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
    st.title("Custom Hitter Leaderboard")
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
#  Constants
# ─────────────────────────────────────────────

from h_utils import STAT_ALLOWLIST,  SUM_STATS, RATE_STATS, format_stat, start_year, MAX_STATS
from h_utils import get_headshot, label_map, lower_better
from h_utils import POSITION_OPTIONS, TEAM_OPTIONS, normalize_team, get_team_display




MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year


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
#  Aggregation (multi-year span)
# ─────────────────────────────────────────────

def apply_dh_override(df):
    if "Pos" not in df.columns or "PA" not in df.columns or "Inn" not in df.columns:
        return df
    df = df.copy()
    pa  = pd.to_numeric(df["PA"],  errors="coerce").fillna(0)
    inn = pd.to_numeric(df["Inn"], errors="coerce").fillna(0)
    estimated = (pa / 4.1) * 9
    is_dh = (inn == 0) | ((inn > 0) & (estimated / inn > 3))
    df.loc[is_dh, "Pos"] = "DH"
    return df

def filter_by_position(df, position):
    df = apply_dh_override(df)
    if position == "all" or "Pos" not in df.columns:
        return df
    if position == "OF":
        return df[df["Pos"].astype(str).str.upper().isin(["LF", "CF", "RF"])]
    return df[df["Pos"].astype(str).str.upper() == position.upper()]

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

    # Team display across seasons
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

    # Recompute counting-derived rate stats
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


# ─────────────────────────────────────────────
#  Main data builder
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
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

    # MODE_MULTI: aggregate by PlayerId
    if "PlayerId" not in combined.columns:
        return combined
    
    combined = filter_by_position(combined, position)
    if combined.empty:
        return pd.DataFrame

    grouped_rows = []
    for _, grp in combined.groupby("PlayerId"):
        grouped_rows.append(aggregate_player_group(grp))

    return pd.DataFrame(grouped_rows)
    

# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

from utils import get_dynamic_min_pa

min_pa = get_dynamic_min_pa(current_year)


for key, default in [
    ("hl_year", current_year),
    ("hl_start_year", current_year - 1),
    ("hl_end_year", current_year),
    ("hl_stat", "fWAR"),
    ("hl_min_pa", min_pa),
    ("hl_position", "all"),
    ("hl_team", "all"),
    ("hl_mode", MODE_SINGLE),
    ("hl_show_player_pa", False),
    ("hl_sort_worst", False),
    ("hl_show_min_pa", True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Controls
# ─────────────────────────────────────────────

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

    st.number_input("Min PA", min_value=0, max_value=20000, key="hl_min_pa")

    st.selectbox("Position", options=list(POSITION_OPTIONS.keys()),
                 format_func=lambda x: POSITION_OPTIONS[x], key="hl_position")

    team_disabled = (mode == MODE_MULTI)
    st.selectbox("Team", options=list(TEAM_OPTIONS.keys()),
                 format_func=lambda x: TEAM_OPTIONS[x], key="hl_team",
                 disabled=team_disabled,
                 help="Team filter unavailable for multi-year span" if team_disabled else None)

    st.checkbox("Show worst",     key="hl_sort_worst")
    st.checkbox("Show min PA",    key="hl_show_min_pa")
    st.checkbox("Show player PA", key="hl_show_player_pa")

min_pa_val   = int(st.session_state.get("hl_min_pa", 0))
position_val = st.session_state.get("hl_position", "all")
team_val     = "all" if team_disabled else st.session_state.get("hl_team", "all")


# ─────────────────────────────────────────────
#  Load & filter data
# ─────────────────────────────────────────────

df = load_data(start_year, end_year, mode, position_val)
if df is None or df.empty:
    st.error(f"No data found for {start_year}–{end_year}.")
    st.stop()

# Min PA filter
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
    player_pa_html = (
        f'<div class="player-pa">{int(pa_val)} PA</div>'
        if st.session_state.get("hl_show_player_pa") and pd.notna(pa_val) else ""
    )

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

# ─────────────────────────────────────────────
#  Title
# ─────────────────────────────────────────────

span_label = f"{start_year}" if mode == MODE_SINGLE else f"{start_year}–{end_year}"
title_label = label_map.get(stat, stat)
pos_suffix = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
team_label = TEAM_OPTIONS.get(team_val, "") if team_val != "all" else ""
mode_label = " Single Season" if mode == MODE_SPLIT else ""
worst_label = "Worst " if sort_worst else ""

title = re.sub(r"  +", " ", f"{span_label}{mode_label} {team_label} {worst_label}{title_label} Leaders{pos_suffix}".strip())


min_pa_subtitle = (
    f'<div class="leaderboard-subtitle">Min {min_pa_val} PA</div>'
    if st.session_state.get("hl_show_min_pa") else ""
)


# ─────────────────────────────────────────────
#  Render HTML
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
html, body {{ background: transparent; font-family: "Source Sans Pro", sans-serif; margin:0; padding:0; }}
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
.leaderboard-subtitle{{
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    margin-top: -1.5rem;
}}
.players-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    justify-content: start;
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
.player-name {{ font-weight: 800; margin-top: 0.35rem; font-size: 1.3rem; }}
.player-team {{ color: #666; font-size: 0.85rem; }}
.player-stat {{ font-weight: 900; font-size: 1.3rem; margin-top: 0.25rem; }}
.player-pa {{ color: #666; font-size: 1rem; }}
html, body {{ margin: 0; padding: 0; background: transparent; width: 100%; }}
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
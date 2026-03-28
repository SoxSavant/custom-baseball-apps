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

TEAM_ALIASES = {"ATH": "OAK", "ATH/OAK": "OAK", "OAK/ATH": "OAK"}

HEADSHOT_BASE = "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current"
HEADSHOT_PLACEHOLDER = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0nMjQwJyBoZWlnaHQ9JzI0MCcgdmlld0JveD0nMCAwIDI0MCAyNDAnIHhtbG5zPSdodHRwOi8v"
    "d3d3LnczLm9yZy8yMDAwL3N2Zyc+CjxyZWN0IHdpZHRoPScyNDAnIGhlaWdodD0nMjQwJyBmaWxsPScjZWVmJy8+"
    "CjxjaXJjbGUgY3g9JzEyMCcgY3k9Jzk1JyByPSc1NScgZmlsbD0nI2RkZScvPgo8Y2lyY2xlIGN4PScxMjAnIGN5"
    "PSc4NScgcj0nNDInIGZpbGw9JyNmZmYnIHN0cm9rZT0nI2NjYycvPgo8cGF0aCBkPSdNMTIwIDE1MGMtMzAgMC01"
    "NSAyNS01NSA1NXMzNSAxNS41IDU1IDE1LjUgNTUtMTUuNSA1NS0xNS41LTM1LTU1LTU1LTU1eicgZmlsbD0nI2Nj"
    "YycvPgo8L3N2Zz4="
)

STAT_ALLOWLIST = [
    "Off", "Def", "BsR", "WAR", "Barrel%", "HardHit%", "EV",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "TB", "H", "1B", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "WPA", "Clutch",
    "Pull%", "GB%", "FB%", "LD%",
    "FRV", "OAA", "DRS",
]

label_map = {
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
    "EV": "Avg Exit Velo",
    "O-Swing%": "Chase%",
}

lower_better = {"K%", "O-Swing%", "SO", "GB%"}

SUM_STATS = {
    "G", "PA", "AB", "R", "H", "1B", "2B", "3B", "HR", "RBI", "SB", "CS",
    "BB", "IBB", "SO", "HBP", "SF", "SH", "XBH", "TB",
    "WAR", "Off", "Def", "BsR", "DRS", "OAA", "FRV",
}
RATE_STATS = {
    "AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "BABIP", "ISO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Barrel%", "HardHit%",
    "Pull%", "GB%", "FB%", "LD%", "EV", "WPA", "Clutch", "wRC+",
}

POSITION_OPTIONS = {
    "all": "All Positions",
    "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
    "LF": "LF", "CF": "CF", "RF": "RF", "OF": "OF", "DH": "DH",
}

TEAM_OPTIONS = {
    "all": "All Teams",
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "CHW": "CHW", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "ATH": "ATH",
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
}

MODE_SINGLE = "Single Season"
MODE_SPLIT  = "Split Seasons"
MODE_MULTI  = "Multi-Year Span"

current_year = date.today().year


# ─────────────────────────────────────────────
#  Team helpers
# ─────────────────────────────────────────────

def normalize_team(team: str) -> str:
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)


def get_team_display(team_value: str) -> str:
    t = str(team_value).strip()
    if t == "- - -":
        return "2+ Teams"
    return normalize_team(t)


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

    # DefPos: most common across seasons
    if "DefPos" in grp.columns:
        pos_vals = grp["DefPos"].dropna().astype(str)
        if not pos_vals.empty:
            result["DefPos"] = pos_vals.mode().iloc[0]

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
def load_data(start_year: int, end_year: int, mode: str) -> pd.DataFrame:
    """
    Single Season: load one year directly.
    Split Seasons: load each year separately, keep as individual rows.
    Multi-Year Span: load all years, aggregate by PlayerId.
    """
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
        # One row per player per season — already the case since each CSV is one season
        return combined

    # MODE_MULTI: aggregate across years by PlayerId
    if "PlayerId" not in combined.columns:
        return combined

    grouped_rows = []
    for pid, grp in combined.groupby("PlayerId"):
        row = aggregate_player_group(grp)
        grouped_rows.append(row)

    return pd.DataFrame(grouped_rows)


# ─────────────────────────────────────────────
#  Headshot
# ─────────────────────────────────────────────

def get_headshot(row: pd.Series) -> str:
    for col in ["MLBAMID", "mlbamid", "mlbam_id", "MLBID"]:
        val = row.get(col)
        if val is not None and pd.notna(val):
            try:
                return HEADSHOT_BASE.format(mlbam=int(val))
            except Exception:
                pass
    return HEADSHOT_PLACEHOLDER


# ─────────────────────────────────────────────
#  Formatting
# ─────────────────────────────────────────────

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()
    if upper_stat in {"FRV", "OAA", "DRS"}:
        return f"{int(round(float(val)))}"
    if upper_stat in {"WAR", "BWAR", "OFF", "DEF", "BSR", "EV"}:
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    if upper_stat in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"
    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO"}:
        return f"{float(val):.3f}".lstrip("0")
    if upper_stat in {"WRC+", "OPS+"}:
        return f"{int(round(float(val)))}"
    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"
    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

for key, default in [
    ("hl_year", current_year-1),
    ("hl_start_year", current_year - 2),
    ("hl_end_year", current_year-1),
    ("hl_stat", "WAR"),
    ("hl_min_pa", 502),
    ("hl_position", "all"),
    ("hl_team", "all"),
    ("hl_mode", MODE_SINGLE),
    ("hl_show_player_pa", False),
    ("hl_sort_worst", False),
    ("hl_show_min_pa", False),
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
        st.number_input("Year", min_value=2015, max_value=current_year-1, key="hl_year")
        start_year = st.session_state["hl_year"]
        end_year   = st.session_state["hl_year"]
    else:
        st.number_input("Start Year", min_value=2015, max_value=current_year-1,
                        value=st.session_state.get("hl_start_year", current_year - 1), key="hl_start_year")
        st.number_input("End Year", min_value=2015, max_value=current_year-1,
                        value=st.session_state.get("hl_end_year", current_year), key="hl_end_year")
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

df = load_data(start_year, end_year, mode)

if df is None or df.empty:
    st.error(f"No data found for {start_year}–{end_year}.")
    st.stop()

# Min PA filter
if min_pa_val > 0 and "PA" in df.columns:
    df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa_val]

# Position filter
if position_val != "all" and "DefPos" in df.columns:
    pos_group = {
        "OF": ["LF", "CF", "RF", "OF"],
    }
    allowed_pos = pos_group.get(position_val, [position_val])
    df = df[df["DefPos"].astype(str).str.upper().isin([p.upper() for p in allowed_pos])]

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

title = re.sub(r"  +", " ", f"{span_label}{mode_label} {team_label} {title_label} Leaders{pos_suffix}".strip())
if sort_worst:
    title += " (Worst)"
if st.session_state.get("hl_show_min_pa"):
    title += f" (min {min_pa_val} PA)"

# ─────────────────────────────────────────────
#  Render HTML
# ─────────────────────────────────────────────

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    <div class="players-grid">{''.join(cards)}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p></p>
        <p>Data: FanGraphs</p>
    </div>
</div>
"""

full_html = f"""
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800&display=swap" rel="stylesheet">
<meta charset="utf-8" />
<style>
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
.players-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    justify-content: start;
    justify-items: center;
    row-gap: 2.5rem;
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
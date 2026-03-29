import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import unicodedata
import html
import re
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Hitting Year-over-Year", layout="wide", page_icon="⚾")

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
    st.title("Year-over-Year Hitter Risers & Fallers")
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

ALLTIME_CSV = Path(__file__).with_name("yoy_deltas.csv")
ALLTIME_MIN_PA = 600

STAT_ALLOWLIST = [
    "Off", "Def", "BsR", "WAR", "bWAR", "Barrel%", "HardHit%", "EV",  "O-Swing%", "Contact%",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "TB", "H", "1B", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "WPA", "Clutch",
    "FRV", "OAA", "DRS", "FRM"
]

ALLTIME_STAT_ALLOWLIST = [
    "WAR", "Off", "Def", "BsR",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG",
    "OPS", "SLG", "OBP", "AVG", "ISO", "BABIP",
    "G", "AB", "R", "RBI", "HR", "SB", "BB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Whiff%",
    "Barrel%", "HardHit%", "EV",
    "GB%", "FB%", "LD%", "Pull%",
    "WPA", "Clutch",
]

label_map = {
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
    "EV": "Avg Exit Velo",
    "O-Swing%": "Chase%",
    "Contact%": "Whiff%",
}

lower_better = {"K%", "O-Swing%", "SO", "Contact%"}

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

current_year = date.today().year


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def normalize_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\xa0", " ").strip()
    try:
        cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(cleaned.split()).lower()


def normalize_team(team: str) -> str:
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)


def get_team_display(team_value: str) -> str:
    t = str(team_value).strip()
    if t == "- - -":
        return "2+ Teams"
    return normalize_team(t)


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


# ─────────────────────────────────────────────
#  YoY delta builder
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_risers_data(
    start_year: int, end_year: int,
    min_pa: int = 0, position: str = "all", team: str = "all"
) -> pd.DataFrame:
    df_s = load_final_year(start_year)
    df_e = load_final_year(end_year)

    if "Contact%" in df_s.columns:
        df_s["Contact%"] = (100 - df_s["Contact%"]*100 )/100
    
    if "Contact%" in df_e.columns:
        df_e["Contact%"] = (100 - df_e["Contact%"]*100)/100 

    if df_s is None or df_s.empty or df_e is None or df_e.empty:
        return pd.DataFrame()

    # Min PA filter
    if min_pa > 0:
        df_s = df_s[pd.to_numeric(df_s.get("PA", 0), errors="coerce").fillna(0) >= min_pa]
        df_e = df_e[pd.to_numeric(df_e.get("PA", 0), errors="coerce").fillna(0) >= min_pa]
    # position filter
    df_s = filter_by_position(df_s, position)
    df_e = filter_by_position(df_e, position)




    # Team filter on end year
    if team != "all" and "Team" in df_e.columns:
        target = normalize_team(team)
        df_e = df_e[df_e["Team"].astype(str).apply(normalize_team) == target]

    # Match on PlayerId
    if "PlayerId" not in df_s.columns or "PlayerId" not in df_e.columns:
        return pd.DataFrame()

    df_s = df_s.set_index("PlayerId")
    df_e = df_e.set_index("PlayerId")
    common_ids = df_s.index.intersection(df_e.index)

    if len(common_ids) == 0:
        return pd.DataFrame()

    df_s = df_s.loc[common_ids]
    df_e = df_e.loc[common_ids]

    skip = {"Season", "Name", "Team", "MLBAMID", "NameASCII", "Pos"}
    numeric_cols = [
        c for c in df_e.columns
        if c not in skip
        and pd.api.types.is_numeric_dtype(df_e[c])
        and c in df_s.columns
    ]

    rows = []
    for pid in common_ids:
        row_s = df_s.loc[pid]
        row_e = df_e.loc[pid]

        record = {
            "PlayerId": pid,
            "Name": row_e.get("Name", row_s.get("Name", "")),
            "Team": get_team_display(str(row_e.get("Team", "N/A"))),
            "PA_start": pd.to_numeric(row_s.get("PA", np.nan), errors="coerce"),
            "PA_end": pd.to_numeric(row_e.get("PA", np.nan), errors="coerce"),
        }

        # Carry MLBAMID from end year
        for col in ["MLBAMID", "mlbamid"]:
            val = row_e.get(col)
            if val is not None and pd.notna(val):
                record[col] = val
                break

        for col in numeric_cols:
            s_val = pd.to_numeric(row_s.get(col, np.nan), errors="coerce")
            e_val = pd.to_numeric(row_e.get(col, np.nan), errors="coerce")
            record[f"{col}_start"] = s_val
            record[f"{col}_end"] = e_val
            record[col] = e_val - s_val

        rows.append(record)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  Stat formatting
# ─────────────────────────────────────────────

def format_stat(stat: str, val, show_sign: bool = False) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FRV", "OAA", "DRS"}:
        v = int(round(float(val)))
        return f"+{v}" if show_sign and v > 0 else f"{v}"

    if upper_stat in {"WAR", "BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR"}:
        v = float(val)
        formatted = f"{abs(v):.1f}" if abs(v - round(v)) >= 1e-9 else f"{int(round(abs(v)))}.0"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"WPA", "CLUTCH"}:
        v = float(val)
        return f"+{v:.2f}" if show_sign and v > 0 else f"{v:.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO"}:
        v = float(val)
        formatted = f"{abs(v):.3f}".lstrip("0") or "0"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"WRC+", "OPS+"}:
        v = int(round(float(val)))
        return f"+{v}" if show_sign and v > 0 else f"{v}"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "Contact" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        formatted = f"{abs(v):.1f}%"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{abs(v):.1f}%" if v < 0 else formatted

    v = float(val)
    formatted = f"{abs(v):.0f}" if abs(v - round(v)) < 1e-6 else f"{abs(v):.1f}"
    if show_sign and v > 0:
        return f"+{formatted}"
    return f"-{formatted}" if v < 0 else formatted


# ─────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────

for key, default in [
    ("rf_start_year",     2024),
    ("rf_end_year",       2025),
    ("rf_stat",           "WAR"),
    ("rf_min_pa",         300),
    ("rf_position",       "all"),
    ("rf_team",           "all"),
    ("rf_show_fallers",   False),
    ("rf_show_min_pa",    True),
    ("rf_show_player_pa", False),
    ("rf_alltime_mode",   False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  Controls
# ─────────────────────────────────────────────



active_allowlist =  STAT_ALLOWLIST
if st.session_state.get("rf_stat") not in active_allowlist:
    st.session_state["rf_stat"] = "WAR"

stat = st.selectbox(
    "Stat", active_allowlist, key="rf_stat",
    format_func=lambda x: label_map.get(x, x),
)

col1, col2 = st.columns([0.5, 2])

with col1:
  
    st.number_input("Start Year", min_value=2015, max_value=current_year, key="rf_start_year")
    st.number_input("End Year",   min_value=2015, max_value=current_year, key="rf_end_year")
    start_year = st.session_state["rf_start_year"]
    end_year   = st.session_state["rf_end_year"]
    if end_year <= start_year:
        st.warning("End Year must be greater than Start Year.")

    st.number_input("Min PA (each year)", min_value=0, max_value=20000, key="rf_min_pa")
    st.selectbox("Position", options=list(POSITION_OPTIONS.keys()),
                    format_func=lambda x: POSITION_OPTIONS[x], key="rf_position")
    st.selectbox("Team", options=list(TEAM_OPTIONS.keys()),
                     format_func=lambda x: TEAM_OPTIONS[x], key="rf_team",
                     help="Filters by team in the end year only")

    min_pa_val   = int(st.session_state.get("rf_min_pa", 0))
    position_val = st.session_state.get("rf_position", "all")
    team_val     = st.session_state.get("rf_team", "all")

    st.checkbox("Show Fallers",   key="rf_show_fallers")
    st.checkbox("Show min PA",    key="rf_show_min_pa")
    st.checkbox("Show player PA", key="rf_show_player_pa")

show_fallers = st.session_state.get("rf_show_fallers", False)

# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────


if end_year > start_year:
    with st.spinner("Loading data..."):
        df = load_risers_data(start_year, end_year, min_pa_val, position_val, team_val)
else:
    df = pd.DataFrame()



# Sort & filter direction
if not df.empty and stat in df.columns:
    stat_lower = stat in lower_better
    ascending  = (stat_lower and not show_fallers) or (not stat_lower and show_fallers)
    df = df.sort_values(by=stat, ascending=ascending)
    numeric_delta = pd.to_numeric(df[stat], errors="coerce")
    if stat_lower:
        df = df[numeric_delta < 0] if not show_fallers else df[numeric_delta > 0]
    else:
        df = df[numeric_delta > 0] if not show_fallers else df[numeric_delta < 0]
    df = df.head(10)
elif not df.empty:
    st.error(f"Stat '{stat}' not found in dataset.")
    df = pd.DataFrame()

print(df)




# ─────────────────────────────────────────────
#  Build cards
# ─────────────────────────────────────────────

cards = []
for _, row in df.iterrows():
    name  = str(row.get("Name", "")).strip()
    team  = str(row.get("Team", ""))
    delta = row.get(stat, np.nan)

    is_positive = pd.notna(delta) and float(delta) > 0
    display_val = format_stat(stat, delta, show_sign=is_positive)

    end_val = row.get(f"{stat}_end", np.nan)
    end_display = format_stat(stat, end_val) if pd.notna(end_val) else ""
    stat_label = label_map.get(stat, stat)

    pa_start = row.get("PA_start", np.nan)
    pa_end   = row.get("PA_end",   np.nan)
    player_pa_html = ""
    if st.session_state.get("rf_show_player_pa"):
        parts = []
        if pd.notna(pa_start): parts.append(str(int(pa_start)))
        if pd.notna(pa_end):   parts.append(str(int(pa_end)))
        if parts:
            player_pa_html = f'<div class="player-pa">{" → ".join(parts)} PA</div>'

    if stat in lower_better:
        is_improvement = pd.notna(delta) and float(delta) < 0
    else:
        is_improvement = is_positive
    delta_class = "stat-positive" if is_improvement else "stat-negative"
    end_context = f'<div class="player-endval">{stat_label}: {end_display}</div>' if end_display else ""

    src = get_headshot(row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(name)}"/>'
    cards.append(f"""
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(name)}</div>
      <div class="player-team">{html.escape(team)}</div>
      <div class="player-stat {delta_class}">{html.escape(display_val)}</div>
      {end_context}
      {player_pa_html}
    </div>
    """)

# ─────────────────────────────────────────────
#  Title
# ─────────────────────────────────────────────

title_stat_label = label_map.get(stat, stat)
pos_suffix  = f" ({POSITION_OPTIONS.get(position_val, '')})" if  position_val != "all" else ""
team_prefix = f"{TEAM_OPTIONS.get(team_val, '')} " if team_val != "all" else ""
riser_label = "Fallers" if show_fallers else "Risers"


title = f"Top {team_prefix}{title_stat_label} {riser_label}{pos_suffix}: {int(start_year)} → {int(end_year)}"

min_pa_subtitle = ""
if st.session_state.get("rf_show_min_pa"):
    display_min =  min_pa_val
    min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {display_min} PA each year</div>'

# ─────────────────────────────────────────────
#  Render HTML
# ─────────────────────────────────────────────

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {min_pa_subtitle}
    <div class="players-grid">
        {''.join(cards) if cards else '<div style="padding:2rem;color:#999;">No data found. Try adjusting your filters or years.</div>'}
    </div>
    <div class="footer">
        <p>By: Sox_Savant</p>
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
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto;
    width: 100%;
    max-width: 900px;
    box-sizing: border-box;
    font-family: "Source Sans Pro", sans-serif;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.2rem;
    margin-bottom: 1rem;
    text-align: center;
}}
.leaderboard-subtitle {{
    text-align: center;
    color: #888;
    font-size: 1.2rem;
    margin-bottom: 1rem;
    margin-top: -0.5rem;
}}
.players-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2.5rem 1rem;
}}
.player-card {{
    flex: 0 0 145px;
    width: 145px;
    text-align: center;
}}
.player-card img {{
    width: 145px;
    height: 145px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    background: #f6f6f6;
}}
.player-name {{ font-weight: 800; margin-top: 0.35rem; font-size: 1.1rem; }}
.player-team {{ color: #666; font-size: 0.85rem; }}
.player-stat {{ font-weight: 900; font-size: 1.5rem; margin-top: 0.25rem; }}
.stat-positive {{ color: #1a7a3c; }}
.stat-negative {{ color: #c0392b; }}
.player-endval {{ color: #888; font-size: .9rem; margin-top: 0.1rem; }}
.player-pa {{ color: #666; font-size: 0.9rem; margin-top: 0.1rem; }}
html, body {{ margin: 0; padding: 0; background: transparent; width: 100%; }}
.footer {{ display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem; }}
.footer p {{ margin: 0; font-size: 0.9rem; color: #666; flex: 1; text-align: center; }}
.footer p:first-child {{ text-align: left; }}
.footer p:last-child  {{ text-align: right; }}
</style>
</head>
<body>{grid_html}</body>
</html>
"""

with col2:
    components.html(full_html, height=850)
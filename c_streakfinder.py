import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

import h_utils
import p_utils


st.set_page_config(page_title="Streak Finder", layout="wide", page_icon="⚾")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem !important;
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

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Streak Finder")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

type_mode = st.radio("Type", ["Hitting", "Pitching"], horizontal=True, key="sf_mode", label_visibility="collapsed")
is_hitting = (type_mode == "Hitting")
U = h_utils if is_hitting else p_utils
prefix = "sfh" if is_hitting else "sfp"

current_year = date.today().year
last_updated = U.get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

STAT_ALLOWLIST = U.STAT_ALLOWLIST
label_map = U.label_map
lower_better = U.lower_better
start_year = U.start_year
STAT_ROUND = U.STAT_ROUND
STAT_DISPLAY_NAMES = U.STAT_DISPLAY_NAMES
load_final_year = U.load_final_year

if is_hitting:
    from h_utils import get_team_display
else:
    from p_utils import get_team_display

default_stat = "xwOBA" if is_hitting else "xERA"
if default_stat not in STAT_ALLOWLIST:
    default_stat = STAT_ALLOWLIST[0]

for key, default in [
    (f"{prefix}_start_year", current_year - 3),
    (f"{prefix}_end_year", current_year),
    (f"{prefix}_stat", default_stat),
    (f"{prefix}_min_pa", 300),
    (f"{prefix}_min_ip", 100),
    (f"{prefix}_direction", "Decreasing"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    start_options = list(range(current_year, start_year - 1, -1))
    current_start_val = st.session_state.get(f"{prefix}_start_year", current_year - 4)
    st.selectbox(
        "Start Year",
        options=start_options,
        index=start_options.index(current_start_val) if current_start_val in start_options else 0,
        key=f"{prefix}_start_year",
    )
    st.selectbox("End Year", options=list(range(current_year, start_year - 1, -1)), key=f"{prefix}_end_year")

with col2:
    stat = st.selectbox(
        "Stat", STAT_ALLOWLIST, key=f"{prefix}_stat",
        format_func=lambda x: label_map.get(x, x),
    )
    direction = st.selectbox(
        "Direction", ["Increasing", "Decreasing"], key=f"{prefix}_direction"
    )

with col3:
    if is_hitting:
        st.number_input("Min PA (each year)", min_value=0, max_value=20000, key=f"{prefix}_min_pa")
    else:
        st.number_input("Min IP (each year)", min_value=0, max_value=5000, key=f"{prefix}_min_ip")

sel_start = st.session_state[f"{prefix}_start_year"]
sel_end = max(st.session_state[f"{prefix}_end_year"], sel_start)
years = list(range(sel_start, sel_end + 1))

if len(years) < 2:
    st.warning("Pick at least a 2-year range.")
    st.stop()

min_pa_val = int(st.session_state.get(f"{prefix}_min_pa", 0))
min_ip_val = int(st.session_state.get(f"{prefix}_min_ip", 0))

stat_lower_better = stat in lower_better


@st.cache_data(show_spinner=False)
def load_year_filtered(year: int, is_hitting: bool,
                        min_pa_val: int, min_ip_val: int) -> pd.DataFrame:
    df = load_final_year(year)
    if df is None or df.empty:
        return pd.DataFrame()

    if is_hitting:
        if min_pa_val > 0 and "PA" in df.columns:
            df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa_val]
    else:
        if min_ip_val > 0 and "IP" in df.columns:
            df = df[pd.to_numeric(df["IP"], errors="coerce").fillna(0) >= min_ip_val]

    return df


def build_streak_table(years, stat, direction):
    per_year = {}
    id_col = None

    for yr in years:
        df = load_year_filtered(yr, is_hitting, min_pa_val, min_ip_val)
        if df.empty or stat not in df.columns or "Name" not in df.columns:
            per_year[yr] = pd.DataFrame(columns=["Name", stat, "Team"]).set_index("Name")
            continue

        if id_col is None:
            for candidate in ("PlayerId", "MLBAMID", "playerid"):
                if candidate in df.columns:
                    id_col = candidate
                    break

        df = df.copy()
        df[stat] = pd.to_numeric(df[stat], errors="coerce")
        df = df.dropna(subset=[stat])

        key_col = id_col if id_col else "Name"
        df = df.drop_duplicates(subset=key_col, keep="first")
        per_year[yr] = df.set_index(key_col)[[stat, "Team", "Name"]]

    common_players = set(per_year[years[0]].index)
    for yr in years[1:]:
        common_players &= set(per_year[yr].index)

    decimals = STAT_ROUND.get(stat, 3)

    rows = []
    for key in common_players:

        values = [round(per_year[yr].loc[key, stat], decimals) for yr in years]
        team_latest = per_year[years[-1]].loc[key, "Team"]
        name_latest = per_year[years[-1]].loc[key, "Name"]
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]

        increasing = all(d > 0 for d in diffs)
        decreasing = all(d < 0 for d in diffs)

        qualifies = increasing if direction == "Increasing" else decreasing
        if not qualifies:
            continue

        row = {"Name": name_latest, "Team": get_team_display(str(team_latest))}
        for yr, v in zip(years, values):
            row[str(yr)] = v
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["_delta"] = result[str(years[-1])] - result[str(years[0])]
    result = result.sort_values("_delta", ascending=(direction != "Increasing"))
    return result.drop(columns="_delta").reset_index(drop=True)


with st.spinner("Scanning years..."):
    table = build_streak_table(years, stat, direction)

if table.empty:
    st.info("No qualified players.")
else:
    st.success(f"{len(table)} player found" if len(table) == 1 else f"{len(table)} players found")

    year_cols = [str(y) for y in years]
    stat_label = STAT_DISPLAY_NAMES.get(stat, label_map.get(stat, stat))
    decimals = STAT_ROUND.get(stat, 1)

    if is_hitting:
        st.markdown(
            f"<div style='text-align:center; margin-bottom: 1rem; color:#888; font-size:0.85rem;'>"
            f"{direction} in {stat_label} each year from {sel_start} to {sel_end} – Min {min_pa_val} PA each year"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:center;  margin-bottom: 1rem; color:#888; font-size:0.85rem;'>"
            f"{direction} in {stat_label} each year from {sel_start} to {sel_end} – Min {min_ip_val} IP each year"
            f"</div>",
            unsafe_allow_html=True,
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            yr: st.column_config.NumberColumn(yr, format=f"%.{decimals}f")
            for yr in year_cols
        },
    )

st.markdown(
    "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem;'>"
    "Data: Baseball Reference · FanGraphs · Baseball Savant"
    "</div>",
    unsafe_allow_html=True,
)
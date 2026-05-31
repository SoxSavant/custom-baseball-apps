import streamlit as st
import pandas as pd
from datetime import date
from zoneinfo import ZoneInfo
from h_utils import (
    load_final_year,
    POSITION_OPTIONS, TEAM_OPTIONS, normalize_team,
    apply_dh_override, s3, bucket,
    STAT_ALLOWLIST, STAT_ROUND, SUM_STATS, MAX_STATS,
    STAT_PRESETS_DATABASE, STAT_DISPLAY_NAMES,
)

st.set_page_config(page_title="Hitting Database", layout="wide", page_icon="⚾")

st.markdown("""
<style>
[data-testid="stToolbar"]      { visibility: hidden; }
[data-testid="stDecoration"]   { display: none; }
[data-testid="stStatusWidget"] { display: none; }
.viewerBadge_link__qRi_k       { display: none; }
.block-container { padding-top: 1rem !important; padding-bottom: 3rem !important; }
.stSelectbox div[data-baseweb="select"],
.stNumberInput > div { max-width: 200px; }
</style>
""", unsafe_allow_html=True)

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
    st.title("Hitting Database")
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
START_YEAR   = 1901

MULTI_TEAM_PLACEHOLDERS = {"---", "TOT", "2TM", "3TM", "4TM", "- - -", ""}


def get_last_updated(year: int) -> str:
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"processed/hitting_final_{year}.csv")
        lm  = obj["LastModified"].astimezone(ZoneInfo("America/New_York"))
        return lm.strftime("%B %d, %Y")
    except Exception:
        return "unknown"

def load_year_range(start: int, end: int) -> pd.DataFrame:
    frames = []
    for yr in range(start, end + 1):
        df = load_final_year(yr)
        if df is not None and not df.empty:
            df = df.copy()
            df["Year"] = yr
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def aggregate_multi_year(df: pd.DataFrame, stat_cols: list, group_cols: list) -> pd.DataFrame:
    """
    Aggregate over multiple seasons.
    - Counting / sum stats → summed
    - Rate stats           → PA-weighted mean
    - Max stats            → max
    - PA                   → summed
    """
    agg: dict = {}

    if "PA" in df.columns:
        df["PA"] = pd.to_numeric(df["PA"], errors="coerce").fillna(0)
        agg["PA"] = ("PA", "sum")

    for stat in stat_cols:
        if stat not in df.columns:
            continue
        df[stat] = pd.to_numeric(df[stat], errors="coerce")

        if stat in SUM_STATS:
            agg[stat] = (stat, "sum")
        elif stat in MAX_STATS:
            agg[stat] = (stat, "max")
        else:
            df[f"_w_{stat}"] = df[stat] * df.get("PA", pd.Series(1, index=df.index))
            agg[f"_w_{stat}"] = (f"_w_{stat}", "sum")

    grouped = df.groupby(group_cols, as_index=False).agg(**agg)

    pa_col = grouped["PA"] if "PA" in grouped.columns else pd.Series(1, index=grouped.index)
    for stat in stat_cols:
        if stat not in df.columns:
            continue
        if stat not in SUM_STATS and stat not in MAX_STATS:
            wkey = f"_w_{stat}"
            if wkey in grouped.columns:
                grouped[stat] = grouped[wkey] / pa_col.replace(0, float("nan"))
                grouped.drop(columns=[wkey], inplace=True)

    return grouped

def fmt(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    decimals = STAT_ROUND.get(stat)
    if decimals is None:
        return str(val)
    if decimals == 0:
        return f"{int(round(float(val)))}"
    return f"{float(val):.{decimals}f}"

from utils import get_dynamic_min_pa

last_updated = get_last_updated(current_year)
st.caption(f"{current_year} data last updated: {last_updated}")

view_mode = st.radio("View", ["Player", "Team"], horizontal=True, label_visibility="collapsed")

c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 1, 2])

with c1:
    year_mode = st.selectbox("Year Mode", ["Single Season", "Multi-Year Span", "Split Season"])

with c2:
    if year_mode == "Single Season":
        year_single = st.selectbox("Year", list(range(current_year, START_YEAR - 1, -1)))
        year_start = year_end = year_single
    else:
        year_start = st.selectbox("From", list(range(current_year, START_YEAR - 1, -1)), index=4)

with c3:
    if year_mode != "Single Season":
        valid_ends = list(range(current_year, year_start - 1, -1))
        year_end = st.selectbox("To", valid_ends, index=0)
    else:
        st.empty()

with c4:
    if view_mode == "Player":
        position = st.selectbox(
            "Position",
            options=list(POSITION_OPTIONS.keys()),
            format_func=lambda x: POSITION_OPTIONS[x],
        )
    else:
        position = "all"
        st.empty()

with c5:
    if view_mode == "Player":
        team = st.selectbox(
            "Team",
            options=list(TEAM_OPTIONS.keys()),
            format_func=lambda x: TEAM_OPTIONS[x],
        )
    else:
        team = "all"
        st.empty()

with c6:
    if view_mode == "Player":
        min_pa = st.number_input("Min PA", min_value=0, max_value=5000, value= get_dynamic_min_pa(current_year), step=10)
    else:
        min_pa = 0

with c7:
    if view_mode == "Player":
        search = st.text_input("Search players (separate by commas)")
    else:
        search = ""

pc1, pc2 = st.columns([1, 3])

with pc1:
    preset_choice = st.selectbox("Stat Preset", STAT_PRESETS_DATABASE)

preset_stats = [s for s in STAT_PRESETS_DATABASE.get(preset_choice, []) if s in STAT_ALLOWLIST]

with pc2:
    selected_stats = st.multiselect(
        "Stats",
        options=STAT_ALLOWLIST,
        default=preset_stats,
        format_func=lambda x: STAT_DISPLAY_NAMES.get(x, x),
    )

if not selected_stats:
    st.warning("Select at least one stat.")
    st.stop()

if year_start == year_end:
    df = load_final_year(year_start)
    if df is not None and not df.empty:
        df["Year"] = year_start
else:
    df = load_year_range(year_start, year_end)

if df is None or df.empty:
    st.error("No data found for the selected year(s).")
    st.stop()

df["Pos"] = df["Pos"].astype(str).str.strip().str.upper()
df = apply_dh_override(df)

multi_year = (year_start != year_end) and (year_mode == "Multi-Year Span")

if view_mode == "Player":
    if position != "all":
        pos = position.upper()
        if pos == "OF":
            df = df[df["Pos"].isin({"LF", "CF", "RF", "OF"})]
        else:
            df = df[df["Pos"] == pos]

    if team != "all" and "Team" in df.columns:
        target = normalize_team(team)
        df = df[df["Team"].astype(str).apply(lambda t: normalize_team(t) == target)]

    if multi_year:
        sorted_df = df.sort_values("Year")
        last = sorted_df.groupby("PlayerId", as_index=False).last()[
            ["PlayerId", "Name", "Team", "Pos"]
        ]
        df = aggregate_multi_year(df, selected_stats, ["PlayerId"])
        df = df.merge(last, on="PlayerId", how="left")

    if min_pa > 0 and "PA" in df.columns:
        df = df[pd.to_numeric(df["PA"], errors="coerce").fillna(0) >= min_pa]

    if search.strip():
        terms = [t.strip() for t in search.split(",") if t.strip()]
        if terms:
            mask = pd.Series(False, index=df.index)
            for term in terms:
                mask |= df["Name"].astype(str).str.contains(term, case=False, na=False)
            df = df[mask]

    if year_mode == "Split Season":
        base_cols = [c for c in ["Name", "Team", "Year", "Pos"] if c in df.columns]
    else:
        base_cols = [c for c in ["Name", "Team", "Pos"] if c in df.columns]

    default_sort = next((s for s in selected_stats if s in df.columns), None)

elif view_mode == "Team":
    if "Team" in df.columns:
        df = df[~df["Team"].astype(str).str.strip().str.upper().isin(MULTI_TEAM_PLACEHOLDERS)]

        if year_mode == "Split Season":
            group_cols = ["Team", "Year"]
        else:
            group_cols = ["Team"]

        df = aggregate_multi_year(df, selected_stats, group_cols)
    else:
        st.error("No Team column found in data.")
        st.stop()

    if year_mode == "Split Season":
        base_cols = ["Team", "Year"]
    else:
        base_cols = ["Team"]

    if "PA" in df.columns:
        base_cols.append("PA")

    default_sort = next((s for s in selected_stats if s in df.columns), None)

seen = set()
deduped_stats = []
for s in selected_stats:
    if s not in seen:
        seen.add(s)
        deduped_stats.append(s)
stat_cols  = [s for s in deduped_stats if s in df.columns]
avail_base = [c for c in base_cols if c in df.columns]
stat_cols  = [s for s in stat_cols if s not in avail_base]
display    = df[avail_base + stat_cols].copy()

if default_sort and default_sort in display.columns:
    display = display.sort_values(default_sort, ascending=False, na_position="last")

display = display.reset_index(drop=True)
display.index += 1

from h_utils import format_stat

PCT_STATS = {
    s for s in stat_cols
    if "%" in s or s in {
        "K%", "BB%", "Chase%", "Whiff%", "Swing%", "Z-Swing%",
        "O-Contact%", "Z-Contact%", "Zone%", "Barrel%", "HardHit%",
        "Sweet-Spot%", "Squared-Up%", "Z-Swing% - Chase%",
    }
}

for stat in PCT_STATS:
    if stat in display.columns:
        display[stat] = display[stat].apply(lambda v: format_stat(stat, v))

col_config: dict = {}
for stat in stat_cols:
    label = STAT_DISPLAY_NAMES.get(stat, stat)
    if stat in PCT_STATS:
        col_config[stat] = st.column_config.TextColumn(label=label)
    else:
        decimals = STAT_ROUND.get(stat, 1)
        col_config[stat] = st.column_config.NumberColumn(
            label=label,
            format=f"%.{decimals}f",
        )

year_label = (
    str(year_start) if year_mode == "Single Season" else f"{year_start}–{year_end}"
)
if view_mode == "Team":
    mode_label = "Team Seasons" if year_mode == "Split Season" else "Teams"
else:
    mode_label = "Player Seasons" if year_mode == "Split Season" else "Players"

st.caption(f"Showing {len(display)} {mode_label} · {year_label}")

st.dataframe(display, use_container_width=True, height=620, column_config=col_config)

st.markdown(
    "<div style='text-align:center; color:#888; font-size:1rem; margin-top:1rem;'>"
    "Data: Baseball Reference · FanGraphs · Baseball Savant"
    "</div>",
    unsafe_allow_html=True,
)
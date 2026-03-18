import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import html
import requests
import re
from bs4 import BeautifulSoup
from pybaseball.statcast_fielding import statcast_outs_above_average
import io
from pathlib import Path
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.set_page_config(page_title="Custom Hitting Leaderboard", layout="wide", page_icon="⚾",)

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;} 
        .viewerBadge_link__qRi_k {display: none;}
        div.ag-header-cell[col-id="ag-RowSelector"],
        div.ag-pinned-left-cols-container [col-id="ag-RowSelector"],
        div.ag-center-cols-container [col-id="ag-RowSelector"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
#  MEMORY-OPTIMIZED DATA LOADING
# ----------------------------

@st.cache_data(ttl=3600, max_entries=3)
def load_filtered_data(start_year, end_year, min_pa=0, position="all"):

    def get_primary_fielding(year_start, year_end, batting_df=None):
        """Get one row per player from fielding stats - their primary position."""
        fielding = pybaseball.fielding_stats(year_start, year_end, qual=0)
        if fielding is None or fielding.empty:
            return pd.DataFrame()
        if "Inn" in fielding.columns:
            total_inn = fielding.groupby("IDfg")["Inn"].sum().rename("TotalInn")
        else:
            total_inn = pd.Series(dtype=float)

        if "Inn" in fielding.columns:
            fielding = fielding.sort_values("Inn", ascending=False)
        fielding = fielding.drop_duplicates(subset=["IDfg"], keep="first")
        fielding = fielding[["IDfg", "Pos"]].rename(columns={"Pos": "DefPos"})
        fielding = fielding.join(total_inn, on="IDfg")

        if batting_df is not None and "PA" in batting_df.columns:
            pa_per_player = (
                batting_df[batting_df["Team"] == "TOT"].set_index("IDfg")["PA"].combine_first(
                    batting_df.drop_duplicates("IDfg").set_index("IDfg")["PA"]
                )
            )
            for fg_id, pa in pa_per_player.items():
                estimated_total_inn = (float(pa) / 4.1) * 9
                field_inn = fielding.loc[fielding["IDfg"] == fg_id, "TotalInn"].values
                field_inn = float(field_inn[0]) if len(field_inn) > 0 else 0
                if field_inn == 0 or (estimated_total_inn / field_inn) > 3:
                    if fg_id in fielding["IDfg"].values:
                        fielding.loc[fielding["IDfg"] == fg_id, "DefPos"] = "DH"
                    else:
                        fielding = pd.concat(
                            [fielding, pd.DataFrame([{"IDfg": fg_id, "DefPos": "DH", "TotalInn": 0}])],
                            ignore_index=True,
                        )
        return fielding[["IDfg", "DefPos", "TotalInn"]] if "TotalInn" in fielding.columns else fielding[["IDfg", "DefPos"]]

    def build_team_display_map(df: pd.DataFrame, year: int) -> dict:
        team_map = {}
        for fg_id, grp in df.groupby("IDfg"):
            raw_teams = grp["Team"].dropna().astype(str).str.strip().str.upper().tolist()
            teams = [normalize_team_code(t, year) for t in raw_teams if t not in {"TOT", "---", "--", "-", ""}]
            teams = sorted(set(t for t in teams if t))
            if teams:
                team_map[fg_id] = compute_team_display(teams)
            else:
                team_map[fg_id] = "2+ Teams"
        return team_map

    # ----------------------------
    #  SINGLE YEAR
    # ----------------------------
    if start_year == end_year:
        df = batting_stats(start_year, end_year, qual=min_pa, split_seasons=False)
        if df.empty:
            return pd.DataFrame()

        if "Team" in df.columns:
            df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")
            df["TeamDisplay"] = df["IDfg"].map(build_team_display_map(df, start_year))

        fielding = get_primary_fielding(start_year, start_year, batting_df=df)
        if not fielding.empty:
            df = df.merge(fielding, on="IDfg", how="left")

        if "H" in df.columns and "Hits" not in df.columns:
            df["Hits"] = df["H"]

        for col in ["H", "2B", "3B", "HR"]:
            if col not in df.columns:
                df[col] = np.nan
        _2b = pd.to_numeric(df["2B"], errors="coerce")
        _3b = pd.to_numeric(df["3B"], errors="coerce")
        _hr = pd.to_numeric(df["HR"], errors="coerce")
        _h  = pd.to_numeric(df["H"],  errors="coerce")
        df["XBH"] = _2b.fillna(0) + _3b.fillna(0) + _hr.fillna(0)
        _1b = _h - _2b - _3b - _hr
        df["TB"] = (_1b.fillna(0) + 2 * _2b.fillna(0) + 3 * _3b.fillna(0) + 4 * _hr.fillna(0)).where(
            _h.notna() & _2b.notna() & _3b.notna() & _hr.notna(), other=np.nan
        )

        if position != "all" and "DefPos" in df.columns:
            pos_values = POSITION_FILTER_MAP.get(position, [])
            df["DefPos"] = df["DefPos"].astype(str).str.upper()
            df = df[df["DefPos"].isin([p.upper() for p in pos_values])]

        return df

    # ----------------------------
    #  MULTI YEAR (aggregated)
    # ----------------------------
    frames = []
    num_years = end_year - start_year + 1
    pre_filter_pa = max(1, min_pa // (num_years * 2)) if min_pa > 0 else 0

    for year in range(start_year, end_year + 1):
        yr_data = batting_stats(year, year, qual=pre_filter_pa, split_seasons=False)
        if yr_data.empty:
            continue

        fielding = get_primary_fielding(year, year, batting_df=yr_data)
        if not fielding.empty:
            yr_data = yr_data.merge(fielding, on="IDfg", how="left")

        tot_ids = set(yr_data.loc[yr_data["Team"] == "TOT", "IDfg"])
        has_individual = set(yr_data.loc[yr_data["Team"] != "TOT", "IDfg"])
        tot_only_ids = tot_ids - has_individual

        non_tot = yr_data[yr_data["Team"] != "TOT"]
        tot_fallback = yr_data[(yr_data["Team"] == "TOT") & (yr_data["IDfg"].isin(tot_only_ids))]
        yr_data = pd.concat([non_tot, tot_fallback], ignore_index=True)
        yr_data = yr_data[yr_data["Team"].notna()]

        if position != "all" and "DefPos" in yr_data.columns:
            pos_values = POSITION_FILTER_MAP.get(position, [])
            yr_data["DefPos"] = yr_data["DefPos"].astype(str).str.upper()
            yr_data = yr_data[yr_data["DefPos"].isin([p.upper() for p in pos_values])]

        if not yr_data.empty:
            yr_data["Season"] = year
            frames.append(yr_data)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = optimize_dtypes(combined)

    grouped_rows = []
    combined = combined[combined["IDfg"].notna()]
    for player_id, grp in combined.groupby("IDfg"):
        name = grp["Name"].iloc[0] if not grp.empty else None
        row = aggregate_player_group(grp, name)
        if row is not None and len(row):
            grouped_rows.append(row)

    result = pd.DataFrame(grouped_rows)
    result = optimize_dtypes(result)

    if not result.empty and min_pa > 0:
        result = result[pd.to_numeric(result.get("PA", 0), errors="coerce").fillna(0) >= min_pa]

    return result


# ----------------------------
#  SPLIT SEASON DATA LOADING
# ----------------------------

@st.cache_data(ttl=3600, max_entries=3)
def load_split_season_data(start_year, end_year, min_pa=0, position="all"):
    """
    Returns one row per player-season (not aggregated across years).
    Used when 'Split seasons' is active so the leaderboard shows the
    best individual season within the span.
    """

    def get_primary_fielding_year(year, batting_df=None):
        fielding = pybaseball.fielding_stats(year, year, qual=0)
        if fielding is None or fielding.empty:
            return pd.DataFrame()
        if "Inn" in fielding.columns:
            total_inn = fielding.groupby("IDfg")["Inn"].sum().rename("TotalInn")
            fielding = fielding.sort_values("Inn", ascending=False)
        else:
            total_inn = pd.Series(dtype=float)
        fielding = fielding.drop_duplicates(subset=["IDfg"], keep="first")
        fielding = fielding[["IDfg", "Pos"]].rename(columns={"Pos": "DefPos"})
        fielding = fielding.join(total_inn, on="IDfg")
        if batting_df is not None and "PA" in batting_df.columns:
            pa_per_player = (
                batting_df[batting_df["Team"] == "TOT"].set_index("IDfg")["PA"].combine_first(
                    batting_df.drop_duplicates("IDfg").set_index("IDfg")["PA"]
                )
            )
            for fg_id, pa in pa_per_player.items():
                estimated_total_inn = (float(pa) / 4.1) * 9
                field_inn = fielding.loc[fielding["IDfg"] == fg_id, "TotalInn"].values
                field_inn = float(field_inn[0]) if len(field_inn) > 0 else 0
                if field_inn == 0 or (estimated_total_inn / field_inn) > 3:
                    if fg_id in fielding["IDfg"].values:
                        fielding.loc[fielding["IDfg"] == fg_id, "DefPos"] = "DH"
                    else:
                        fielding = pd.concat(
                            [fielding, pd.DataFrame([{"IDfg": fg_id, "DefPos": "DH", "TotalInn": 0}])],
                            ignore_index=True,
                        )
        return fielding[["IDfg", "DefPos", "TotalInn"]] if "TotalInn" in fielding.columns else fielding[["IDfg", "DefPos"]]

    frames = []
    for year in range(start_year, end_year + 1):
        yr_data = batting_stats(year, year, qual=min_pa, split_seasons=False)
        if yr_data is None or yr_data.empty:
            continue

        fielding = get_primary_fielding_year(year, batting_df=yr_data)
        if not fielding.empty:
            yr_data = yr_data.merge(fielding, on="IDfg", how="left")

        # Resolve TOT rows: prefer individual team rows
        tot_ids = set(yr_data.loc[yr_data["Team"] == "TOT", "IDfg"])
        has_individual = set(yr_data.loc[yr_data["Team"] != "TOT", "IDfg"])
        tot_only_ids = tot_ids - has_individual
        non_tot = yr_data[yr_data["Team"] != "TOT"]
        tot_fallback = yr_data[(yr_data["Team"] == "TOT") & (yr_data["IDfg"].isin(tot_only_ids))]
        yr_data = pd.concat([non_tot, tot_fallback], ignore_index=True)
        yr_data = yr_data[yr_data["Team"].notna()]

        # Collapse multi-team players to one row per player for this season
        yr_data["IDfg"] = pd.to_numeric(yr_data["IDfg"], errors="coerce")
        collapsed = []
        for fg_id, grp in yr_data.groupby("IDfg"):
            raw_teams = grp["Team"].dropna().astype(str).str.strip().str.upper().tolist()
            teams = [normalize_team_code(t, year) for t in raw_teams if t not in {"TOT", "---", "--", "-", ""}]
            teams = sorted(set(t for t in teams if t))
            team_display = compute_team_display(teams) if teams else "2+ Teams"

            tot_row = grp[grp["Team"] == "TOT"]
            if not tot_row.empty:
                base = tot_row.iloc[0].to_dict()
            else:
                base = grp.iloc[0].to_dict()

            base["TeamDisplay"] = team_display
            base["Season"] = year
            collapsed.append(base)

        yr_data = pd.DataFrame(collapsed)

        # Position filter
        if position != "all" and "DefPos" in yr_data.columns:
            pos_values = POSITION_FILTER_MAP.get(position, [])
            yr_data["DefPos"] = yr_data["DefPos"].astype(str).str.upper()
            yr_data = yr_data[yr_data["DefPos"].isin([p.upper() for p in pos_values])]

        # Derived stats
        if "H" in yr_data.columns and "Hits" not in yr_data.columns:
            yr_data["Hits"] = yr_data["H"]
        for col in ["H", "2B", "3B", "HR"]:
            if col not in yr_data.columns:
                yr_data[col] = np.nan
        _2b = pd.to_numeric(yr_data["2B"], errors="coerce")
        _3b = pd.to_numeric(yr_data["3B"], errors="coerce")
        _hr = pd.to_numeric(yr_data["HR"], errors="coerce")
        _h  = pd.to_numeric(yr_data["H"],  errors="coerce")
        yr_data["XBH"] = _2b.fillna(0) + _3b.fillna(0) + _hr.fillna(0)
        _1b = _h - _2b - _3b - _hr
        yr_data["TB"] = (_1b.fillna(0) + 2*_2b.fillna(0) + 3*_3b.fillna(0) + 4*_hr.fillna(0)).where(
            _h.notna() & _2b.notna() & _3b.notna() & _hr.notna(), other=np.nan
        )

        if not yr_data.empty:
            frames.append(yr_data)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return optimize_dtypes(combined)


def optimize_dtypes(df):
    if df.empty:
        return df
    df = df.copy()
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        if col not in ["AVG", "OBP", "SLG", "wOBA", "xwOBA", "xBA", "xSLG"]:
            df[col] = df[col].astype("float32")
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        if df[col].max() < 2147483647:
            df[col] = df[col].astype("int32")
    return df


def normalize_statcast_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    cleaned = name.replace("\xa0", " ").strip()
    if "," in cleaned:
        last, first = cleaned.split(",", 1)
        full = f"{first.strip()} {last.strip()}"
    else:
        full = cleaned
    try:
        full = unicodedata.normalize("NFKD", full).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(full.split())


def normalize_team_code(team: str, year: int) -> str | None:
    if not team:
        return None
    team = team.upper().strip()
    if team in {"", "-", "--", "---", "- - -", "TOT"}:
        return None
    if year < 2025:
        if team in {"ATH", "OAK"}:
            return "OAK"
    else:
        if team in {"ATH", "OAK"}:
            return "ATH"
    return team


def collapse_athletics(teams: list[str]) -> list[str]:
    has_oak = "OAK" in teams
    has_ath = "ATH" in teams
    if has_oak and has_ath:
        new_list = [t for t in teams if t not in {"OAK", "ATH"}]
        new_list.append("OAK/ATH")
        return sorted(new_list)
    return teams


VALID_TEAMS = {
    "ARI","ATL","BAL","BOS","CHC","CIN","CLE","COL","CHW","DET",
    "HOU","KCR","LAA","LAD","MIA","MIL","MIN","NYM","NYY",
    "OAK","ATH","PHI","PIT","SDP","SEA","SFG","STL","TBR",
    "TEX","TOR","WSN"
}

LOCAL_BWAR_FILE = Path(__file__).with_name("warhitters2025.txt")


def compute_team_display(teams: list[str]) -> str:
    if not teams:
        return "2+ Teams"
    if len(teams) == 1:
        return teams[0]
    return "2+ Teams"


# ----------------------------
#  External Data Loaders
# ----------------------------

POSITION_FILTER_MAP = {
    "all": None,
    "C":   ["C"],
    "1B":  ["1B"],
    "2B":  ["2B"],
    "3B":  ["3B"],
    "SS":  ["SS"],
    "LF": ["LF"],
    "CF": ["CF"],
    "RF": ["RF"],
    "OF":  ["LF", "CF", "RF", "OF"],
    "DH":  ["DH"],
}


@st.cache_data(ttl=600, max_entries=10)
def batting_stats(start_year: int, end_year: int, qual=0, split_seasons=False):
    try:
        df = pybaseball.batting_stats(start_year, end_year, qual=qual, split_seasons=split_seasons)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600, max_entries=10)
def load_savant_frv_year(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/fielding-run-value?"
        f"gameType=Regular&seasonStart={year}&seasonEnd={year}"
        "&type=fielder&position=&minInnings=0&minResults=1&csv=true"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = io.StringIO(resp.content.decode("utf-8"))
        df = pd.read_csv(data)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"name": "NameRaw", "total_runs": "FRV", "arm_runs": "ARM", "range_runs": "RANGE"})
    
    df["Name"] = df["NameRaw"].astype(str).str.strip()
    df["NameKey"] = df["Name"].apply(normalize_statcast_name)
    for metric in ["FRV", "ARM", "RANGE"]:
        df[metric] = pd.to_numeric(df.get(metric), errors="coerce")
    return df[["NameKey", "Name", "FRV", "ARM", "RANGE"]]


@st.cache_data(show_spinner=False, ttl=600, max_entries=10)
def load_savant_oaa_year(year: int) -> pd.DataFrame:
    try:
        df = statcast_outs_above_average(year, "all")
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    name_col = None
    for col in ["player_name", "last_name, first_name", "name"]:
        if col in df.columns:
            name_col = col
            break
    if not name_col:
        return pd.DataFrame()
    if name_col == "last_name, first_name":
        df["Name"] = df[name_col].apply(lambda x: (str(x) or "").strip())
    else:
        df["Name"] = df[name_col].astype(str).str.strip()
    df["NameKey"] = df["Name"].apply(normalize_statcast_name)
    oaa_col = None
    for col in ["outs_above_average", "oaa"]:
        if col in df.columns:
            oaa_col = col
            break
    if not oaa_col:
        return pd.DataFrame()
    df["OAA"] = pd.to_numeric(df[oaa_col], errors="coerce")
    return df[["NameKey", "Name", "OAA"]]


@st.cache_data(show_spinner=False, ttl=600, max_entries=5)
def load_fangraphs_fielding(player_names: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    if not player_names:
        return pd.DataFrame()
    try:
        df = pybaseball.fielding_stats(start_year, end_year, qual=0, split_seasons=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df["NameKey"] = df["Name"].apply(normalize_statcast_name)
        target_keys = set([normalize_statcast_name(n) for n in player_names])
        df = df[df["NameKey"].isin(target_keys)]
        if df.empty:
            return pd.DataFrame()
        result = df.groupby("NameKey", as_index=False).agg({"DRS": "sum", "TZ": "sum", "UZR": "sum", "FRM": "sum"})
        return result
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False, max_entries=5)
def load_fielding_for_players(player_names: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    if not player_names:
        return pd.DataFrame()
    target_keys = set([normalize_statcast_name(n) for n in player_names])
    frames = []
    for year in range(start_year, end_year + 1):
        frv = load_savant_frv_year(year)
        if frv is not None and not frv.empty:
            frv = frv[frv["NameKey"].isin(target_keys)]
            if not frv.empty:
                frv["Season"] = year
                frames.append(frv)
        oaa = load_savant_oaa_year(year)
        if oaa is not None and not oaa.empty:
            oaa = oaa[oaa["NameKey"].isin(target_keys)]
            if not oaa.empty:
                oaa["Season"] = year
                frames.append(oaa)
    savant_data = pd.DataFrame()
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["NameKey"] = combined["NameKey"].astype(str)
        agg_cols = {c: "sum" for c in ["FRV", "ARM", "RANGE", "OAA"] if c in combined.columns}
        savant_data = combined.groupby("NameKey", as_index=False).agg(agg_cols)
    fangraphs_data = load_fangraphs_fielding(player_names, start_year, end_year)
    if not savant_data.empty and not fangraphs_data.empty:
        result = savant_data.merge(fangraphs_data, on="NameKey", how="outer")
    elif not savant_data.empty:
        result = savant_data
    elif not fangraphs_data.empty:
        result = fangraphs_data
    else:
        result = pd.DataFrame()
    return result


HEADSHOT_BASES = [
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current",
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current",
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}headshot/67/current",
]
HEADSHOT_BREF_BASES = [
    "https://content-static.baseball-reference.com/req/202406/images/headshots/{folder}/{bref_id}.jpg",
    "https://content-static.baseball-reference.com/req/202310/images/headshots/{folder}/{bref_id}.jpg",
    "https://www.baseball-reference.com/req/202108020/images/headshots/{folder}/{bref_id}.jpg",
]
HEADSHOT_CHECK_TIMEOUT = 1.0
HEADSHOT_USER_AGENT = "headshot-fetcher/1.0"
HEADSHOT_PLACEHOLDER = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0nMjQwJyBoZWlnaHQ9JzI0MCcgdmlld0JveD0nMCAwIDI0MCAyNDAnIHhtbG5zPSdodHRwOi8v"
    "d3d3LnczLm9yZy8yMDAwL3N2Zyc+CjxyZWN0IHdpZHRoPScyNDAnIGhlaWdodD0nMjQwJyBmaWxsPScjZWVmJy8+"
    "CjxjaXJjbGUgY3g9JzEyMCcgY3k9Jzk1JyByPSc1NScgZmlsbD0nI2RkZScvPgo8Y2lyY2xlIGN4PScxMjAnIGN5"
    "PSc4NScgcj0nNDInIGZpbGw9JyNmZmYnIHN0cm9rZT0nI2NjYycvPgo8cGF0aCBkPSdNMTIwIDE1MGMtMzAgMC01"
    "NSAyNS01NSA1NXMzNSAxNS41IDU1IDE1LjUgNTUtMTUuNSA1NS0xNS41LTM1LTU1LTU1LTU1eicgZmlsbD0nI2Nj"
    "YycvPgo8L3N2Zz4="
)


@st.cache_data(show_spinner=False)
def lookup_mlbam_id(full_name: str, return_bbref: bool = False):
    if not full_name or not full_name.strip():
        return (None, None) if return_bbref else None
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

    def normalize_token(tok: str) -> str:
        if not tok:
            return ""
        tok = tok.replace(".", "").strip()
        try:
            return unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode()
        except Exception:
            return tok

    def clean_full(val: str) -> str:
        try:
            val = unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
        except Exception:
            pass
        return "".join(ch for ch in val if ch.isalnum()).lower()

    def strip_suffix(tokens: list[str]) -> list[str]:
        toks = tokens.copy()
        while toks and toks[-1].lower() in suffixes:
            toks.pop()
        return toks

    parts = full_name.split()
    base_tokens = strip_suffix(parts)
    if len(base_tokens) < 2:
        return (None, None) if return_bbref else None

    first_raw = base_tokens[0]
    last_raw = " ".join(base_tokens[1:])
    target_clean = clean_full(first_raw + last_raw)

    def initial_forms(token: str) -> list[str]:
        forms = []
        if not token:
            return forms
        stripped = token.replace(".", "")
        if stripped and stripped.isupper() and 1 <= len(stripped) <= 4:
            dotted = ".".join(list(stripped)) + "."
            spaced = " ".join(list(stripped))
            forms.extend([dotted, spaced, stripped, stripped + "."])
        return forms

    first_forms = initial_forms(first_raw)
    variants = [
        (last_raw, first_raw),
        (normalize_token(last_raw), normalize_token(first_raw)),
        (normalize_token(last_raw).lower(), normalize_token(first_raw).lower()),
        (last_raw.replace(".", ""), first_raw.replace(".", "")),
    ]
    for form in first_forms:
        variants.append((last_raw, form))
        variants.append((normalize_token(last_raw), normalize_token(form)))

    first_hit_mlbam = None
    first_hit_bbref = None
    best_match_mlbam = None
    best_match_bbref = None

    def consider_row(row):
        nonlocal first_hit_mlbam, first_hit_bbref, best_match_mlbam, best_match_bbref
        combo = clean_full(str(row.get("name_first", "")) + str(row.get("name_last", "")))
        mlbam_val = row.get("key_mlbam")
        bbref_val = row.get("key_bbref")
        if combo == target_clean:
            if pd.notna(mlbam_val):
                try:
                    best_match_mlbam = int(mlbam_val)
                except Exception:
                    pass
            if pd.notna(bbref_val):
                try:
                    best_match_bbref = str(bbref_val)
                except Exception:
                    pass
        if first_hit_mlbam is None and pd.notna(mlbam_val):
            try:
                first_hit_mlbam = int(mlbam_val)
            except Exception:
                pass
        if first_hit_bbref is None and pd.notna(bbref_val):
            try:
                first_hit_bbref = str(bbref_val)
            except Exception:
                pass

    for last, first in variants:
        try:
            lookup_df = pybaseball.playerid_lookup(last, first)
        except Exception:
            continue
        if lookup_df is None or lookup_df.empty:
            continue
        for _, row in lookup_df.iterrows():
            consider_row(row)

    try:
        lookup_df = pybaseball.playerid_lookup(last_raw, None)
    except Exception:
        lookup_df = None
    if lookup_df is not None and not lookup_df.empty:
        for _, row in lookup_df.iterrows():
            consider_row(row)

    mlbam_result = best_match_mlbam if best_match_mlbam is not None else first_hit_mlbam
    bbref_result = best_match_bbref if best_match_bbref is not None else first_hit_bbref

    if return_bbref:
        return mlbam_result, bbref_result
    return mlbam_result


@st.cache_data(show_spinner=False, ttl=21600)
def build_mlb_headshot(mlbam: int | str | None) -> str | None:
    if mlbam is None:
        return None
    mlbam_val = str(mlbam).strip()
    if not mlbam_val:
        return None
    headers = {"User-Agent": HEADSHOT_USER_AGENT}
    fallback_url = None
    for base in HEADSHOT_BASES:
        try:
            url = base.format(mlbam=mlbam_val)
            if fallback_url is None:
                fallback_url = url
        except Exception:
            continue
        try:
            resp = requests.head(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, allow_redirects=True)
            status = resp.status_code
            if status == 200:
                return url
            if status in (403, 404, 405):
                resp_get = requests.get(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, stream=True)
                if resp_get.status_code == 200:
                    return url
        except Exception:
            continue
    return fallback_url


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def reverse_lookup_mlbam(fg_id: int) -> int | None:
    try:
        rev = pybaseball.playerid_reverse_lookup([int(fg_id)], key_type="fangraphs")
        if rev is not None and not rev.empty:
            mlbam = rev.iloc[0].get("key_mlbam")
            if pd.notna(mlbam):
                return int(mlbam)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_bbref_headshot(bref_id: str | None) -> str | None:
    if not bref_id:
        return None
    slug = str(bref_id).strip().lower()
    if not slug:
        return None
    first_letter = slug[0]
    url = f"https://www.baseball-reference.com/players/{first_letter}/{slug}.shtml"
    headers = {"User-Agent": HEADSHOT_USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT)
    except Exception:
        return None
    if resp.status_code != 200 or not resp.text:
        return None
    html_text = resp.text
    urls = []
    for pattern in [
        r'https?://[^"\']*headshots[^"\']*\.(?:jpg|png)',
        r'//[^"\']*headshots[^"\']*\.(?:jpg|png)',
    ]:
        urls.extend(re.findall(pattern, html_text, flags=re.IGNORECASE))
    for raw in urls:
        if not raw:
            continue
        candidate = raw if raw.startswith("http") else f"https:{raw}"
        return candidate
    return None


def build_bref_headshot(bref_id: str | None) -> str | None:
    if not bref_id:
        return None
    raw_slug = str(bref_id).strip()
    if not raw_slug:
        return None
    slug_variants = {raw_slug.lower(), raw_slug.upper()}
    for slug in slug_variants:
        folder_variants = {slug[0].lower(), slug[0].upper()} if slug else set()
        for folder in folder_variants:
            for base in HEADSHOT_BREF_BASES:
                try:
                    return base.format(folder=folder, bref_id=slug)
                except Exception:
                    continue
    return None


def resolve_bref_headshot(bref_id: str | None) -> str | None:
    direct = build_bref_headshot(bref_id)
    if direct:
        return direct
    return fetch_bbref_headshot(bref_id)


def heuristic_bbref_slug(full_name: str) -> list[str]:
    def clean_name(val: str) -> str:
        if not val:
            return ""
        try:
            val = unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
        except Exception:
            pass
        return "".join(ch for ch in val if ch.isalnum() or ch.isspace()).strip().lower()
    cleaned = clean_name(full_name)
    if not cleaned:
        return []
    parts = cleaned.split()
    if len(parts) < 2:
        return []
    first = parts[0]
    last = parts[-1]
    if not first or not last:
        return []
    base_slug = f"{last[:5]}{first[:2]}"
    if len(base_slug) < 6:
        return []
    slugs = []
    for i in range(1, 16):
        slugs.append(f"{base_slug}{i:02d}")
    return slugs


def get_headshot_url_from_row(row: pd.Series) -> str:
    name = str(row.get("Name", "")).strip()
    id_cols = ["mlbam_override", "mlbamid", "mlbam_id", "mlbam", "MLBID", "MLBAMID", "key_mlbam"]
    for col in id_cols:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                try:
                    mlbam = int(val)
                    headshot = build_mlb_headshot(mlbam)
                    if headshot:
                        return headshot
                except Exception:
                    pass
    fg_cols = ["playerid", "IDfg", "fg_id", "FGID"]
    for col in fg_cols:
        if col in row.index:
            fg = row.get(col)
            if pd.notna(fg) and str(fg).strip():
                try:
                    mlbam = reverse_lookup_mlbam(int(fg))
                    if mlbam:
                        headshot = build_mlb_headshot(mlbam)
                        if headshot:
                            return headshot
                except Exception:
                    pass
    bref_cols = ["key_bbref", "bbref_id", "BBREFID", "bref_id", "BREFID"]
    for col in bref_cols:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                bref_url = resolve_bref_headshot(str(val))
                if bref_url:
                    return bref_url
    if name:
        mlbam_fallback, bbref_fallback = lookup_mlbam_id(name, return_bbref=True)
        if mlbam_fallback:
            headshot = build_mlb_headshot(mlbam_fallback)
            if headshot:
                return headshot
        if bbref_fallback:
            bref_url = resolve_bref_headshot(bbref_fallback)
            if bref_url:
                return bref_url
    if name:
        for slug in heuristic_bbref_slug(name):
            bref_url = resolve_bref_headshot(slug)
            if bref_url:
                return bref_url
    return HEADSHOT_PLACEHOLDER


# ----------------------------
#  Aggregation
# ----------------------------

def aggregate_player_group(grp: pd.DataFrame, name: str | None = None) -> dict:
    result: dict[str, object] = {}

    if name is None and "Name" in grp.columns:
        val = grp["Name"].dropna()
        if not val.empty:
            name = str(val.iloc[0])
    if name:
        result["Name"] = name

    teams = grp.loc[grp["Team"].notna() & (grp["Team"] != "TOT"), "Team"].astype(str).tolist()
    teams = [normalize_team_code(t, int(grp["Season"].iloc[0]) if "Season" in grp.columns else 2025) for t in teams]
    teams = collapse_athletics(sorted(set([t for t in teams if t])))

    if not teams:
        result["TeamDisplay"] = "TOT"
    elif len(teams) == 1:
        result["TeamDisplay"] = teams[0]
    else:
        result["TeamDisplay"] = compute_team_display(teams)

    result["Teams"] = teams

    try:
        if "Season" in grp.columns:
            grp_sorted = grp.sort_values(by="Season", ascending=False)
        else:
            grp_sorted = grp.iloc[::-1]

        mlb_cols = ["mlbam", "MLBID", "key_mlbam", "mlbam_id", "MLBAMID"]
        fg_cols = ["playerid", "IDfg", "fg_id", "FGID"]

        found_mlb = None
        found_fg = None
        for _, r in grp_sorted.iterrows():
            if found_mlb is None:
                for c in mlb_cols:
                    if c in r.index:
                        v = r.get(c)
                        if pd.notna(v) and str(v).strip():
                            try:
                                found_mlb = int(v)
                            except Exception:
                                found_mlb = str(v).strip()
                            break
            if found_fg is None:
                for c in fg_cols:
                    if c in r.index:
                        v = r.get(c)
                        if pd.notna(v) and str(v).strip():
                            try:
                                found_fg = int(v)
                            except Exception:
                                found_fg = str(v).strip()
                            break
            if found_mlb is not None and found_fg is not None:
                break

        if found_mlb is not None:
            result["mlbam"] = found_mlb
        if found_fg is not None:
            result["IDfg"] = found_fg
    except Exception:
        pass

    skip_cols = {
        "Name", "Team", "Season", "Teams",
        "mlbam", "MLBID", "key_mlbam", "mlbam_id", "MLBAMID",
        "playerid", "IDfg", "fg_id", "FGID",
    }

    for col in grp.columns:
        if col in skip_cols:
            continue
        try:
            result[col] = grp[col].sum()
        except Exception:
            result[col] = grp[col].iloc[0]

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

    singles = result.get("1B")
    if singles is None or pd.isna(singles):
        if pd.notna(h) and pd.notna(doubles) and pd.notna(triples) and pd.notna(hr):
            try:
                singles = h - doubles - triples - hr
            except Exception:
                singles = np.nan
    try:
        singles = float(singles) if singles is not None and not pd.isna(singles) else np.nan
    except Exception:
        singles = np.nan

    tb = result.get("TB")
    if tb is None or pd.isna(tb):
        comps = [singles, doubles, triples, hr]
        if all(pd.notna(x) for x in comps):
            tb = singles + 2 * doubles + 3 * triples + 4 * hr
        else:
            tb = np.nan

    if pd.notna(doubles) and pd.notna(triples) and pd.notna(hr):
        result["XBH"] = doubles + triples + hr
    else:
        result["XBH"] = np.nan

    if pd.notna(h):
        result["Hits"] = h

    if pd.notna(ab) and ab > 0 and pd.notna(h):
        result["AVG"] = h / ab
    else:
        result["AVG"] = np.nan

    if pd.notna(ab) and ab > 0 and pd.notna(tb):
        result["SLG"] = tb / ab
    else:
        result["SLG"] = np.nan

    bb_val = 0 if pd.isna(bb) else bb
    hbp_val = 0 if pd.isna(hbp) else hbp
    sf_val = 0 if pd.isna(sf) else sf
    obp_den = (ab if pd.notna(ab) else 0) + bb_val + hbp_val + sf_val
    if obp_den > 0 and pd.notna(h):
        result["OBP"] = (h + bb_val + hbp_val) / obp_den
    else:
        result["OBP"] = np.nan

    if pd.notna(result.get("SLG")) and pd.notna(result.get("AVG")):
        try:
            result["ISO"] = float(result.get("SLG")) - float(result.get("AVG"))
        except Exception:
            result["ISO"] = np.nan
    else:
        result["ISO"] = np.nan

    pa_total = 0.0
    if "PA" in result and pd.notna(result.get("PA")):
        try:
            pa_total = float(result.get("PA"))
        except Exception:
            pa_total = 0.0

    rate_stats = {
        "AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "BABIP",
        "K%", "BB%", "K-BB%", "O-Swing%", "Z-Swing%", "Swing%", "Contact%", "Whiff%",
        "Barrel%", "HardHit%", "Pull%", "Cent%", "Oppo%", "GB%", "FB%", "LD%", "LA",
        "EV", "MaxEV", "CSW%", "BB/K", "ISO", "WRC+"
    }

    if pa_total > 0:
        for rs in rate_stats:
            matching_col = None
            for c in grp.columns:
                try:
                    if str(c).upper() == str(rs).upper():
                        matching_col = c
                        break
                except Exception:
                    continue

            if matching_col is not None:
                try:
                    vals = pd.to_numeric(grp[matching_col], errors="coerce").fillna(np.nan)
                    pas = pd.to_numeric(grp.get("PA", 0), errors="coerce").fillna(0)
                    numer = (vals * pas).sum(skipna=True)
                    if numer is None or np.isnan(numer):
                        result[matching_col] = np.nan
                    else:
                        result[matching_col] = numer / pa_total
                except Exception:
                    result[matching_col] = np.nan
            else:
                if rs in result and pd.notna(result.get(rs)):
                    continue
                result[rs] = np.nan

    return result


# ----------------------------
#  Formatting
# ----------------------------

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""

    upper_stat = stat.upper()
    if upper_stat in {"FRV", "ARM"}:
        return f"{int(round(float(val)))}"

    if upper_stat == "AGE":
        if isinstance(val, str):
            return val
        v = float(val)
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"WAR", "BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR"}:
        v = float(val)
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}.0"
        return f"{v:.1f}"

    if upper_stat in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO"}:
        return f"{float(val):.3f}".lstrip("0")

    if upper_stat in {"WRC+", "OPS+"}:
        return f"{int(round(float(val)))}"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"

    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def transform_stat_value(stat: str, raw_val):
    if stat == "Contact%":
        if pd.isna(raw_val):
            return np.nan
        try:
            contact = float(raw_val)
        except Exception:
            return np.nan
        if contact <= 1:
            contact *= 100
        return 100 - contact
    return raw_val


# ----------------------------
#  UI Constants
# ----------------------------

STAT_ALLOWLIST = [
    "Off", "Def", "BsR", "WAR", "Barrel%", "HardHit%", "EV",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "TB", "Hits", "1B", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Z-Swing%", "Swing%", "Contact%", "WPA", "Clutch",
    "Pull%", "Cent%", "Oppo%", "GB%", "FB%", "LD%", "LA",
    "FRV", "OAA", "ARM", "DRS", "TZ", "UZR", "FRM",
]

label_map = {
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
    "EV": "Avg Exit Velo",
    "Contact%": "Whiff%",
    "O-Swing%": "Chase%",
    "Hits": "Hits",
}

lower_better = {"K%", "O-Swing%", "Contact%", "SO", "GB%"}

POSITION_OPTIONS = {
    "all": "All Positions",
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "LF": "LF",
    "CF": "CF",
    "RF": "RF",
    "OF": "OF",
    "DH": "DH",
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

# Mode constants
MODE_SINGLE   = "Single Season"
MODE_SPLIT    = "Split Seasons"
MODE_MULTI    = "Multi-Year Span"

# ----------------------------
#  Page header
# ----------------------------
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

current_year = date.today().year

# ----------------------------
#  Session state defaults
# ----------------------------
for key, default in [
    ("hl_year", 2025),
    ("hl_start_year", 2024),
    ("hl_end_year", 2025),
    ("hl_stat", "WAR"),
    ("hl_min_pa", 502),
    ("hl_position", "all"),
    ("hl_team", "all"),
    ("hl_mode", MODE_SINGLE),
    ("hl_show_player_pa", False),
    ("hl_show_innings", False),
    ("hl_sort_worst", False),
    ("hl_show_min_pa", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(
    """
    <style>
        .stSelectbox div[data-baseweb="select"],
        .stNumberInput > div {
            max-width: 200px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
#  Controls
# ----------------------------
stat = st.selectbox(
    "Stat",
    STAT_ALLOWLIST,
    key="hl_stat",
    format_func=lambda x: label_map.get(x, x),
)

col1, col2 = st.columns([.5, 2])

with col1:
    # Mode radio — single select, 3 options
    mode = st.radio(
        "Mode",
        options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI],
        key="hl_mode",
    )

    # Year inputs depend on mode
    if mode == MODE_SINGLE:
        st.number_input(
            "Year",
            min_value=1900,
            max_value=current_year,
            key="hl_year",
        )
        start_year = st.session_state["hl_year"]
        end_year   = st.session_state["hl_year"]

    else:  # Split Seasons or Multi-Year Span — both need a range
        st.number_input(
            "Start Year",
            min_value=1900,
            max_value=current_year,
            value=st.session_state.get("hl_start_year", 2024),
            key="hl_start_year",
        )
        st.number_input(
            "End Year",
            min_value=1900,
            max_value=current_year,
             value=st.session_state.get("hl_end_year", 2025),
            key="hl_end_year",
        )
        start_year = st.session_state["hl_start_year"]
        end_year   = st.session_state["hl_end_year"]
        # Guard: end year must be >= start year
        if end_year < start_year:
            end_year = start_year

    st.number_input(
        "Min PA",
        min_value=0,
        max_value=20000,
        key="hl_min_pa",
    )

    st.selectbox(
        "Position",
        options=list(POSITION_OPTIONS.keys()),
        format_func=lambda x: POSITION_OPTIONS[x],
        key="hl_position",
    )

    # Team selector: disabled (greyed out) in Multi-Year Span mode
    team_disabled = (mode == MODE_MULTI)
    st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key="hl_team",
        disabled=team_disabled,
        help="Team filter is not available for multi-year span mode, I had difficulty aggregating players who played for multiple teams or were traded midseason" if team_disabled else None,
    )

    st.checkbox("Show worst",       key="hl_sort_worst")
    st.checkbox("Show min PA",      key="hl_show_min_pa")
    st.checkbox("Show player PA",   key="hl_show_player_pa")
    st.checkbox("Show player innings", key="hl_show_innings")
if stat == "FRV":
    if start_year < 2018:
        st.warning("⚠️ FRV may be understated for catchers before 2018 due to missing framing data from Baseball Savant.")
# Resolved values
min_pa_val    = int(st.session_state.get("hl_min_pa", 0))
position_val  = st.session_state.get("hl_position", "all")
# If team selector was disabled, treat as "all"
team_val      = "all" if team_disabled else st.session_state.get("hl_team", "all")

split_seasons_active = (mode == MODE_SPLIT)
multi_span_active    = (mode == MODE_MULTI)

# ----------------------------
#  Load data
# ----------------------------
if split_seasons_active:
    df = load_split_season_data(start_year, end_year, min_pa_val, position_val)
    if team_val != "all" and not df.empty and "Team" in df.columns:
        df = df[df["Team"].apply(
            lambda v: normalize_team_code(str(v).upper().strip(), start_year) == team_val
            if pd.notna(v) else False
        )]
else:
    # Single season OR multi-year span (no team filter for multi)
    df = load_filtered_data(start_year, end_year, min_pa_val, position_val)
    if team_val != "all" and not df.empty and "Team" in df.columns:
        df = df[df["Team"].apply(
            lambda v: normalize_team_code(str(v).upper().strip(), start_year) == team_val
            if pd.notna(v) else False
        )]

# Slim df to only columns we actually use
if not df.empty:
    keep_cols = set(STAT_ALLOWLIST) | {
        "Name", "Team", "TeamDisplay", "IDfg", "Age", "Season",
        "mlbam", "MLBID", "key_mlbam", "mlbam_id", "DefPos", "TotalInn",
        "H", "TB",
    }
    df = df[[c for c in df.columns if c in keep_cols]]

if not df.empty and stat in ["FRV", "OAA", "ARM", "DRS", "TZ", "UZR", "FRM"]:
    player_names = df["Name"].tolist()
    if split_seasons_active and "Season" in df.columns:
        # Merge fielding per year so split season rows get the correct single-year value
        df["NameKey"] = df["Name"].apply(normalize_statcast_name)
        fielding_frames = []
        for yr in df["Season"].dropna().unique():
            yr = int(yr)
            yr_fielding = load_fielding_for_players(player_names, yr, yr)
            if not yr_fielding.empty:
                yr_fielding["Season"] = yr
                fielding_frames.append(yr_fielding)
        if fielding_frames:
            all_fielding = pd.concat(fielding_frames, ignore_index=True)
            df = df.merge(all_fielding, on=["NameKey", "Season"], how="left", suffixes=("", "_fielding"))
    else:
        fielding_data = load_fielding_for_players(player_names, start_year, end_year)
        if not fielding_data.empty:
            df["NameKey"] = df["Name"].apply(normalize_statcast_name)
            df = df.merge(fielding_data, on="NameKey", how="left", suffixes=("", "_fielding"))
    for col in ["FRV", "OAA", "ARM", "RANGE", "DRS", "TZ", "UZR", "FRM"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

# ----------------------------
#  Sort & take top 10
# ----------------------------
if not df.empty and stat in df.columns:
    stat_is_lower_better = stat in lower_better
    sort_worst = st.session_state.get("hl_sort_worst", False)
    ascending = (stat_is_lower_better and not sort_worst) or (not stat_is_lower_better and sort_worst)
    df = df.sort_values(by=stat, ascending=ascending)
    df = df.head(10)
elif not df.empty:
    st.error(f"Column '{stat}' not found. Available columns: {', '.join(df.columns)}")
    df = pd.DataFrame()

if not df.empty and "TeamDisplay" not in df.columns:
    df["TeamDisplay"] = "2+ Teams"

# ----------------------------
#  Build cards
# ----------------------------
cards = []
for _, row in df.iterrows():
    name     = row.get("Name", "")
    team     = row.get("TeamDisplay", "")
    raw_val  = row.get(stat, np.nan)
    transformed  = transform_stat_value(stat, raw_val)
    display_val  = format_stat(stat, transformed)

    # In split-season mode, append the season year under the team
    if split_seasons_active and "Season" in row.index and pd.notna(row.get("Season")):
        season_yr = int(row["Season"])
        team = f"{team} ({season_yr})"

    src_row = row
    try:
        pos = list(df.index).index(row.name)
        key = f"hl_mlbam_override_{pos}"
        override_val = st.session_state.get(key, "")
        if override_val is not None and str(override_val).strip():
            try:
                ov = int(str(override_val).strip())
                src_row = row.copy()
                src_row["mlbam_override"] = ov
            except Exception:
                pass
    except Exception:
        pass

    pa_val = row.get("PA", np.nan)
    player_pa_display = (
        f'<div class="player-pa">{int(pa_val)} PA</div>'
        if st.session_state.get("hl_show_player_pa", False) and pd.notna(pa_val)
        else ""
    )

    innings_val = row.get("TotalInn", np.nan)
    player_innings_display = (
        f'<div class="player-innings">{int(innings_val)} Inn.</div>'
        if st.session_state.get("hl_show_innings", False) and pd.notna(innings_val) and innings_val > 0
        else ""
    )

    src = get_headshot_url_from_row(src_row)
    img_html  = f'<img src="{html.escape(src)}" alt="{html.escape(str(name))}"/>'
    card_html = f'''
    <div class="player-card">
      {img_html}
      <div class="player-name">{name}</div>
      <div class="player-team">{team}</div>
      <div class="player-stat">{display_val}</div>
      {player_pa_display}
      {player_innings_display}
    </div>
    '''
    cards.append(card_html)

# ----------------------------
#  Title
# ----------------------------
if mode == MODE_SINGLE:
    span_label = f"{int(start_year)}"
else:
    span_label = f"{int(start_year)}\u2013{int(end_year)}"

title_label = label_map.get(stat, stat)
pos_display = POSITION_OPTIONS.get(position_val, "")
pos_suffix  = f" ({pos_display})" if position_val != "all" else ""

team_label = ""
if team_val != "all":
    team_label = TEAM_OPTIONS.get(team_val, "")

mode_label = ""
if mode == MODE_SPLIT:
    mode_label = " Single Season"

title = f"{span_label}{mode_label} {team_label} {title_label} Leaders{pos_suffix}".strip()
# Clean up any double spaces
import re as _re
title = _re.sub(r"  +", " ", title)

if st.session_state.get("hl_sort_worst", False):
    title += " (Worst)"
if st.session_state.get("hl_show_min_pa", False):
    try:
        min_pa_display = int(st.session_state.get("hl_min_pa", 0))
    except Exception:
        min_pa_display = 0
    title += f" (min {min_pa_display} PA)"

FIELDING_STATS = {"FRV", "OAA", "ARM", "DRS", "TZ", "UZR", "FRM"}
footer_middle = ""
if position_val != "all" and stat in FIELDING_STATS:
    pos_label  = POSITION_OPTIONS.get(position_val, position_val)
    stat_label = label_map.get(stat, stat)
    footer_middle = f'<p>Total {stat_label} among primary {pos_label}</p>'

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{title}</div>
    <div class="players-grid">
        {''.join(cards)}
    </div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        {footer_middle}
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
.player-card {{
    text-align: center;
}}
.player-card img {{
    width: 155px;
    height: 155px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    background: #f6f6f6;
}}
.player-name {{
    font-weight: 800;
    margin-top: 0.35rem;
    font-size: 1.3rem;
}}
.player-team {{
    color: #666;
    font-size: 0.85rem;
}}
.player-stat {{
    font-weight: 900;
    font-size: 1.3rem;
    margin-top: 0.25rem;
}}
.player-pa {{
    color: #666;
    font-size: 1rem;
}}
.player-innings {{
    color: #666;
    font-size: 1rem;
}}
html, body {{
    margin: 0px;
    padding: 0px;
    background: transparent;
    width: 100%;
}}
.footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: .5rem;
}}
.footer p {{
    margin: 0;
    font-size: 0.9rem;
    color: #666;
    font-family: "Source Sans Pro";
    flex: 1;
    text-align: center;
}}
.footer p:first-child {{
    text-align: left;
}}
.footer p:last-child {{
    text-align: right;
}}
</style>
</head>
<body>
{grid_html}
</body>
</html>
"""

with col2:
    components.html(full_html, height=800)

# ----------------------------
#  MLBAM overrides
# ----------------------------
if not df.empty:
    st.markdown("---")
    st.write("Manual MLBAM overrides (enter MLBAM id to fix headshot)")

    cols_row1 = st.columns(5)
    for col_idx in range(5):
        player_idx = col_idx
        if player_idx >= len(df):
            break
        idx = df.index[player_idx]
        row = df.loc[idx]
        with cols_row1[col_idx]:
            key = f"hl_mlbam_override_{player_idx}"
            default_val = ""
            if "mlbam_override" in df.columns and pd.notna(row.get("mlbam_override")):
                try:
                    default_val = str(int(row["mlbam_override"]))
                except Exception:
                    default_val = str(row["mlbam_override"]) if pd.notna(row.get("mlbam_override")) else ""
            user_val = st.text_input(f"Player {player_idx+1} MLBAM", value=default_val, key=key)
            try:
                if user_val and str(user_val).strip():
                    df.at[idx, "mlbam_override"] = int(str(user_val).strip())
                else:
                    df.at[idx, "mlbam_override"] = np.nan
            except Exception:
                df.at[idx, "mlbam_override"] = np.nan

    cols_row2 = st.columns(5)
    for col_idx in range(5):
        player_idx = col_idx + 5
        if player_idx >= len(df):
            break
        idx = df.index[player_idx]
        row = df.loc[idx]
        with cols_row2[col_idx]:
            key = f"hl_mlbam_override_{player_idx}"
            default_val = ""
            if "mlbam_override" in df.columns and pd.notna(row.get("mlbam_override")):
                try:
                    default_val = str(int(row["mlbam_override"]))
                except Exception:
                    default_val = str(row["mlbam_override"]) if pd.notna(row.get("mlbam_override")) else ""
            user_val = st.text_input(f"Player {player_idx+1} MLBAM", value=default_val, key=key)
            try:
                if user_val and str(user_val).strip():
                    df.at[idx, "mlbam_override"] = int(str(user_val).strip())
                else:
                    df.at[idx, "mlbam_override"] = np.nan
            except Exception:
                df.at[idx, "mlbam_override"] = np.nan
import requests
import time

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.fangraphs.com/",
    "Connection": "keep-alive"
})



def safe_get(url, **kwargs):
    for _ in range(3):
        try:
            r = SESSION.get(url, timeout=15, **kwargs)
            if r.status_code == 200:
                return r
        except:
            pass
        time.sleep(2)
    return None

requests.get = SESSION.get
requests.post = SESSION.post

import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import html
import re
from bs4 import BeautifulSoup
from pybaseball.statcast_fielding import statcast_outs_above_average
import io
from pathlib import Path
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.error("⚠️ Data source temporarily down. Working on a fix.")

st.set_page_config(page_title="Hitting Year-over-Year", layout="wide", page_icon="⚾",)

POSITION_FILTER_MAP = {
    "all": None,
    "C":   ["C"],
    "1B":  ["1B"],
    "2B":  ["2B"],
    "3B":  ["3B"],
    "SS":  ["SS"],
    "LF":  ["LF"],
    "CF":  ["CF"],
    "RF":  ["RF"],
    "OF":  ["LF", "CF", "RF", "OF"],
    "DH":  ["DH"],
}

POSITION_OPTIONS = {
    "all": "All Positions",
    "C":  "C",  "1B": "1B", "2B": "2B", "3B": "3B",
    "SS": "SS", "LF": "LF", "CF": "CF", "RF": "RF",
    "OF": "OF", "DH": "DH",
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
    "WAR":      "fWAR",
    "EV":       "Avg Exit Velo",
    "Contact%": "Whiff%",
    "O-Swing%": "Chase%",
    "Hits":     "Hits",
}

lower_better = {"K%", "O-Swing%", "Contact%", "SO", "GB%"}

HEADSHOT_BASES = [
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current",
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current",
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

HEADSHOT_BREF_BASES = [
    "https://content-static.baseball-reference.com/req/202406/images/headshots/{folder}/{bref_id}.jpg",
    "https://content-static.baseball-reference.com/req/202310/images/headshots/{folder}/{bref_id}.jpg",
]

FIELDING_STATS = {"FRV", "OAA", "ARM", "DRS", "TZ", "UZR", "FRM"}

# Stats available in the precomputed CSV (no fielding stats)
ALLTIME_STAT_ALLOWLIST = [
    "WAR", "Off", "Def", "BsR",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG",
    "OPS", "SLG", "OBP", "AVG", "ISO", "BABIP",
    "G", "AB", "R", "RBI", "HR", "SB", "BB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Contact%",
    "Barrel%", "HardHit%", "EV",
    "GB%", "FB%", "LD%", "Pull%",
    "WPA", "Clutch",
]

ALLTIME_CSV = Path(__file__).with_name("yoy_deltas.csv")
ALLTIME_MIN_PA = 600


@st.cache_data(show_spinner=False, ttl= 3600)
def load_alltime_csv() -> pd.DataFrame:
    if not ALLTIME_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(ALLTIME_CSV)
    # Convert Contact% columns: stored as raw FanGraphs contact%, convert to whiff%
    for col in [c for c in df.columns if c.startswith("Contact%")]:
        vals = pd.to_numeric(df[col], errors="coerce")
        median_val = vals.median()
        if pd.notna(median_val) and median_val <= 1:
            vals = vals * 100
        df[col] = 100 - vals
    return df


def get_alltime_year_pairs(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    pairs = sorted(df[["start_year", "end_year"]].drop_duplicates().itertuples(index=False, name=None))
    return pairs


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


def format_stat(stat: str, val, show_sign: bool = False) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FRV", "ARM"}:
        v = int(round(float(val)))
        if show_sign and v > 0:
            return f"+{v}"
        return f"{v}"

    if upper_stat == "AGE":
        if isinstance(val, str):
            return val
        v = float(val)
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"WAR", "BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR"}:
        v = float(val)
        formatted = f"{abs(v):.1f}" if abs(v - round(v)) >= 1e-9 else f"{int(round(abs(v)))}.0"
        if show_sign and v > 0:
            return f"+{formatted}"
        elif v < 0:
            return f"-{formatted}"
        return formatted

    if upper_stat in {"WPA", "CLUTCH"}:
        v = float(val)
        if show_sign and v > 0:
            return f"+{v:.2f}"
        return f"{v:.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO"}:
        v = float(val)
        formatted = f"{abs(v):.3f}".lstrip("0") or "0"
        if show_sign and v > 0:
            return f"+.{formatted.lstrip('.')}" if formatted.startswith(".") else f"+{formatted}"
        elif v < 0:
            raw = f"{abs(v):.3f}".lstrip("0") or "0"
            return f"-.{raw.lstrip('.')}" if raw.startswith(".") else f"-{raw}"
        return formatted

    if upper_stat in {"WRC+", "OPS+"}:
        v = int(round(float(val)))
        if show_sign and v > 0:
            return f"+{v}"
        return f"{v}"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        formatted = f"{abs(v):.1f}%"
        if show_sign and v > 0:
            return f"+{formatted}"
        elif v < 0:
            return f"-{abs(v):.1f}%"
        return formatted

    v = float(val)
    formatted = f"{abs(v):.0f}" if abs(v - round(v)) < 1e-6 else f"{abs(v):.1f}"
    if show_sign and v > 0:
        return f"+{formatted}"
    elif v < 0:
        return f"-{formatted}"
    return formatted


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


@st.cache_data(ttl=3600, max_entries=10)
def batting_stats_cached(year: int, qual=0):
    try:
        df = pybaseball.batting_stats(year, year, qual=qual, split_seasons=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, max_entries=10)
def fielding_stats_cached(year: int):
    try:
        df = pybaseball.fielding_stats(year, year, qual=0)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def get_primary_fielding(year: int, batting_df=None) -> pd.DataFrame:
    fielding = fielding_stats_cached(year)
    if fielding is None or fielding.empty:
        return pd.DataFrame()
    if "Inn" in fielding.columns:
        total_inn = fielding.groupby("IDfg")["Inn"].sum().rename("TotalInn")
        fielding = fielding.sort_values("Inn", ascending=False)
    else:
        total_inn = pd.Series(dtype=float)
    fielding = fielding.drop_duplicates(subset=["IDfg"], keep="first")
    fielding = fielding[["IDfg", "Pos"]].rename(columns={"Pos": "DefPos"})
    if len(total_inn):
        fielding = fielding.join(total_inn, on="IDfg")

    if batting_df is not None and "PA" in batting_df.columns:
        pa_per_player = (
            batting_df[batting_df["Team"] == "TOT"].set_index("IDfg")["PA"].combine_first(
                batting_df.drop_duplicates("IDfg").set_index("IDfg")["PA"]
            )
        )
        for fg_id, pa in pa_per_player.items():
            estimated_total_inn = (float(pa) / 4.1) * 9
            field_inn = fielding.loc[fielding["IDfg"] == fg_id, "TotalInn"].values if "TotalInn" in fielding.columns else []
            field_inn = float(field_inn[0]) if len(field_inn) > 0 else 0
            if field_inn == 0 or (estimated_total_inn / field_inn) > 3:
                if fg_id in fielding["IDfg"].values:
                    fielding.loc[fielding["IDfg"] == fg_id, "DefPos"] = "DH"
                else:
                    fielding = pd.concat(
                        [fielding, pd.DataFrame([{"IDfg": fg_id, "DefPos": "DH", "TotalInn": 0}])],
                        ignore_index=True,
                    )
    cols = ["IDfg", "DefPos", "TotalInn"] if "TotalInn" in fielding.columns else ["IDfg", "DefPos"]
    return fielding[cols]


def load_single_year(year: int, min_pa: int = 0, position: str = "all") -> pd.DataFrame:
    df = batting_stats_cached(year, qual=min_pa)
    if df is None or df.empty:
        return pd.DataFrame()

    df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")

    tot_ids = set(df.loc[df["Team"] == "TOT", "IDfg"])
    has_individual = set(df.loc[df["Team"] != "TOT", "IDfg"])
    tot_only_ids = tot_ids - has_individual
    non_tot = df[df["Team"] != "TOT"]
    tot_fallback = df[(df["Team"] == "TOT") & (df["IDfg"].isin(tot_only_ids))]
    df = pd.concat([non_tot, tot_fallback], ignore_index=True)
    df = df[df["Team"].notna()]

    fielding = get_primary_fielding(year, batting_df=df)
    if not fielding.empty:
        df = df.merge(fielding, on="IDfg", how="left")

    collapsed = []
    for fg_id, grp in df.groupby("IDfg"):
        raw_teams = grp["Team"].dropna().astype(str).str.strip().str.upper().tolist()
        teams = [normalize_team_code(t, year) for t in raw_teams if t not in {"TOT", "---", "--", "-", ""}]
        teams = sorted(set(t for t in teams if t))

        tot_row = grp[grp["Team"] == "TOT"]
        base = tot_row.iloc[0].to_dict() if not tot_row.empty else grp.iloc[0].to_dict()

        if len(teams) == 1:
            base["TeamDisplay"] = teams[0]
        elif teams:
            base["TeamDisplay"] = "2+ Teams"
        else:
            base["TeamDisplay"] = "2+ Teams"

        base["_teams_list"] = teams

        if "DefPos" in grp.columns:
            base["DefPos"] = grp.dropna(subset=["DefPos"])["DefPos"].iloc[0] if not grp.dropna(subset=["DefPos"]).empty else base.get("DefPos", "")

        collapsed.append(base)

    df = pd.DataFrame(collapsed)

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

    # Convert Contact% to Whiff% in place so deltas are computed correctly
    if "Contact%" in df.columns:
        contact = pd.to_numeric(df["Contact%"], errors="coerce")
        median_val = contact.median()
        if pd.notna(median_val) and median_val <= 1:
            contact = contact * 100
        df["Contact%"] = 100 - contact

    return df


@st.cache_data(show_spinner=False, ttl=3600, max_entries=10)
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


@st.cache_data(show_spinner=False, ttl=3600, max_entries=10)
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
    oaa_col = next((c for c in ["outs_above_average", "oaa"] if c in df.columns), None)
    if not oaa_col:
        return pd.DataFrame()
    df["OAA"] = pd.to_numeric(df[oaa_col], errors="coerce")
    return df[["NameKey", "Name", "OAA"]]


@st.cache_data(show_spinner=False, ttl=3600, max_entries=5)
def load_fangraphs_fielding(player_names: list, start_year: int, end_year: int) -> pd.DataFrame:
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


def load_fielding_for_year(player_names: list, year: int) -> pd.DataFrame:
    if not player_names:
        return pd.DataFrame()
    target_keys = set([normalize_statcast_name(n) for n in player_names])
    frames = []
    frv = load_savant_frv_year(year)
    if frv is not None and not frv.empty:
        frv = frv[frv["NameKey"].isin(target_keys)]
        if not frv.empty:
            frames.append(frv)
    oaa = load_savant_oaa_year(year)
    if oaa is not None and not oaa.empty:
        oaa = oaa[oaa["NameKey"].isin(target_keys)]
        if not oaa.empty:
            frames.append(oaa)

    savant_data = pd.DataFrame()
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        agg_cols = {c: "sum" for c in ["FRV", "ARM", "RANGE", "OAA"] if c in combined.columns}
        savant_data = combined.groupby("NameKey", as_index=False).agg(agg_cols)

    fg_data = load_fangraphs_fielding(player_names, year, year)

    if not savant_data.empty and not fg_data.empty:
        return savant_data.merge(fg_data, on="NameKey", how="outer")
    elif not savant_data.empty:
        return savant_data
    elif not fg_data.empty:
        return fg_data
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def lookup_mlbam_id(full_name: str, return_bbref: bool = False):
    if not full_name or not full_name.strip():
        return (None, None) if return_bbref else None
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

    def normalize_token(tok):
        if not tok:
            return ""
        tok = tok.replace(".", "").strip()
        try:
            return unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode()
        except Exception:
            return tok

    def clean_full(val):
        try:
            val = unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
        except Exception:
            pass
        return "".join(ch for ch in val if ch.isalnum()).lower()

    def strip_suffix(tokens):
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

    variants = [
        (last_raw, first_raw),
        (normalize_token(last_raw), normalize_token(first_raw)),
        (normalize_token(last_raw).lower(), normalize_token(first_raw).lower()),
        (last_raw.replace(".", ""), first_raw.replace(".", "")),
    ]

    best_match_mlbam = None
    best_match_bbref = None
    first_hit_mlbam = None
    first_hit_bbref = None

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
                best_match_bbref = str(bbref_val)
        if first_hit_mlbam is None and pd.notna(mlbam_val):
            try:
                first_hit_mlbam = int(mlbam_val)
            except Exception:
                pass
        if first_hit_bbref is None and pd.notna(bbref_val):
            first_hit_bbref = str(bbref_val)

    for last, first in variants:
        try:
            lookup_df = pybaseball.playerid_lookup(last, first)
        except Exception:
            continue
        if lookup_df is None or lookup_df.empty:
            continue
        for _, row in lookup_df.iterrows():
            consider_row(row)

    mlbam_result = best_match_mlbam if best_match_mlbam is not None else first_hit_mlbam
    bbref_result = best_match_bbref if best_match_bbref is not None else first_hit_bbref
    if return_bbref:
        return mlbam_result, bbref_result
    return mlbam_result


@st.cache_data(show_spinner=False, ttl=21600)
def build_mlb_headshot(mlbam) -> str | None:
    if mlbam is None:
        return None
    mlbam_val = str(mlbam).strip()
    headers = {"User-Agent": HEADSHOT_USER_AGENT}
    for base in HEADSHOT_BASES:
        try:
            url = base.format(mlbam=mlbam_val)
            resp = requests.head(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200:
                return url
            if resp.status_code in (403, 404, 405):
                resp_get = requests.get(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, stream=True)
                if resp_get.status_code == 200:
                    return url
        except Exception:
            continue
    return HEADSHOT_BASES[0].format(mlbam=mlbam_val)


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


def build_bref_headshot(bref_id: str | None) -> str | None:
    if not bref_id:
        return None
    slug = str(bref_id).strip()
    if not slug:
        return None
    for base in HEADSHOT_BREF_BASES:
        try:
            return base.format(folder=slug[0].lower(), bref_id=slug)
        except Exception:
            continue
    return None


def get_headshot_url_from_row(row: pd.Series) -> str:
    name = str(row.get("Name", "")).strip()
    for col in ["mlbam_override", "mlbamid", "mlbam_id", "mlbam", "MLBID", "MLBAMID", "key_mlbam"]:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                try:
                    headshot = build_mlb_headshot(int(val))
                    if headshot:
                        return headshot
                except Exception:
                    pass
    for col in ["playerid", "IDfg", "fg_id", "FGID"]:
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
    if name:
        mlbam_fallback, bbref_fallback = lookup_mlbam_id(name, return_bbref=True)
        if mlbam_fallback:
            headshot = build_mlb_headshot(mlbam_fallback)
            if headshot:
                return headshot
        if bbref_fallback:
            bref_url = build_bref_headshot(bbref_fallback)
            if bref_url:
                return bref_url
    return HEADSHOT_PLACEHOLDER


def load_risers_data(start_year: int, end_year: int, min_pa: int = 0,
                     position: str = "all", team: str = "all") -> pd.DataFrame:
    df_start = load_single_year(start_year, min_pa=min_pa, position=position)
    df_end   = load_single_year(end_year,   min_pa=min_pa, position=position)

    if df_start.empty or df_end.empty:
        return pd.DataFrame()

    # Team filter: end year only
    if team != "all":
        def played_for_team(teams_list, yr):
            if not teams_list:
                return False
            return any(normalize_team_code(t, yr) == team for t in teams_list)

        if "_teams_list" in df_end.columns:
            df_end = df_end[df_end["_teams_list"].apply(lambda t: played_for_team(t, end_year))]

    df_start["IDfg"] = pd.to_numeric(df_start["IDfg"], errors="coerce")
    df_end["IDfg"]   = pd.to_numeric(df_end["IDfg"],   errors="coerce")

    common_ids = set(df_start["IDfg"].dropna()) & set(df_end["IDfg"].dropna())
    df_start = df_start[df_start["IDfg"].isin(common_ids)].set_index("IDfg")
    df_end   = df_end[df_end["IDfg"].isin(common_ids)].set_index("IDfg")

    if df_start.empty or df_end.empty:
        return pd.DataFrame()

    skip_meta = {"Name", "Team", "TeamDisplay", "_teams_list", "DefPos", "TotalInn",
                 "mlbam", "MLBID", "key_mlbam", "mlbam_id", "Season"}
    numeric_cols = [
        c for c in df_end.columns
        if c not in skip_meta
        and pd.api.types.is_numeric_dtype(df_end[c])
        and c in df_start.columns
    ]

    result_rows = []
    for fg_id in common_ids:
        if fg_id not in df_start.index or fg_id not in df_end.index:
            continue
        row_s = df_start.loc[fg_id]
        row_e = df_end.loc[fg_id]

        record = {
            "IDfg": fg_id,
            "Name": row_e.get("Name", row_s.get("Name", "")),
            "TeamDisplay": row_e.get("TeamDisplay", ""),
            "DefPos": row_e.get("DefPos", ""),
            "TotalInn": row_e.get("TotalInn", np.nan),
        }
        for hcol in ["mlbam", "MLBID", "key_mlbam", "mlbam_id"]:
            if hcol in row_e.index and pd.notna(row_e.get(hcol)):
                record[hcol] = row_e[hcol]
            elif hcol in row_s.index and pd.notna(row_s.get(hcol)):
                record[hcol] = row_s[hcol]

        for col in numeric_cols:
            try:
                s_val = pd.to_numeric(row_s[col] if col in row_s.index else np.nan, errors="coerce")
                e_val = pd.to_numeric(row_e[col] if col in row_e.index else np.nan, errors="coerce")
                record[f"{col}_start"] = s_val
                record[f"{col}_end"]   = e_val
                record[col] = e_val - s_val
            except Exception:
                record[col] = np.nan

        record["PA_start"]      = pd.to_numeric(row_s.get("PA",       np.nan), errors="coerce")
        record["PA_end"]        = pd.to_numeric(row_e.get("PA",       np.nan), errors="coerce")
        record["TotalInn_start"] = pd.to_numeric(row_s.get("TotalInn", np.nan), errors="coerce")
        record["TotalInn_end"]   = pd.to_numeric(row_e.get("TotalInn", np.nan), errors="coerce")

        result_rows.append(record)

    if not result_rows:
        return pd.DataFrame()

    return pd.DataFrame(result_rows)


# ----------------------------
#  PAGE HEADER
# ----------------------------
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

current_year = date.today().year

for key, default in [
    ("rf_start_year",   2024),
    ("rf_end_year",     2025),
    ("rf_stat",         "WAR"),
    ("rf_min_pa",       300),
    ("rf_position",     "all"),
    ("rf_team",         "all"),
    ("rf_show_fallers", False),
    ("rf_show_min_pa",  True),
    ("rf_show_player_pa", False),
    ("rf_show_innings", False),
    ("rf_alltime_mode", False),
    ("rf_min_inn", 0),
    ("rf_show_min_inn", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

alltime_mode = st.checkbox("🕰️ All-Time Mode (600 PA, 1955–present)", key="rf_alltime_mode")

st.markdown(
    """
    <style>
        .stSelectbox div[data-baseweb="select"],
        .stNumberInput > div { max-width: 200px; }
    </style>
    """,
    unsafe_allow_html=True,
)

active_allowlist = ALLTIME_STAT_ALLOWLIST if alltime_mode else STAT_ALLOWLIST
if st.session_state.get("rf_stat") not in active_allowlist:
    st.session_state["rf_stat"] = "WAR"

stat = st.selectbox(
    "Stat",
    active_allowlist,
    key="rf_stat",
    format_func=lambda x: label_map.get(x, x),
)

col1, col2 = st.columns([0.5, 2])

with col1:
    if alltime_mode:
        st.info(f"Showing all-time best/worst single-season changes (min {ALLTIME_MIN_PA} PA each year)")
        min_pa_val   = ALLTIME_MIN_PA
        min_inn_val  = 0
        position_val = "all"
        team_val     = "all"
        start_year   = None
        end_year     = None

    else:
        st.number_input("Start Year", min_value=1900, max_value=current_year, key="rf_start_year")
        st.number_input("End Year",   min_value=1900, max_value=current_year, key="rf_end_year")

        start_year = st.session_state["rf_start_year"]
        end_year   = st.session_state["rf_end_year"]
        if end_year <= start_year:
            st.warning("End Year must be greater than Start Year.")

        st.number_input("Min PA (each year)", min_value=0, max_value=20000, key="rf_min_pa")
        st.number_input("Min Inn (each year)", min_value=0, max_value=10000, key="rf_min_inn", help="Minimum innings played in both years. Useful for filtering out DH-only seasons for fielding stats.")

        st.selectbox(
            "Position",
            options=list(POSITION_OPTIONS.keys()),
            format_func=lambda x: POSITION_OPTIONS[x],
            key="rf_position",
        )

        st.selectbox(
            "Team",
            options=list(TEAM_OPTIONS.keys()),
            format_func=lambda x: TEAM_OPTIONS[x],
            key="rf_team",
            help="Filters by team in the end year only",
        )

        min_pa_val   = int(st.session_state.get("rf_min_pa", 0))
        min_inn_val  = int(st.session_state.get("rf_min_inn", 0))
        position_val = st.session_state.get("rf_position", "all")
        team_val     = st.session_state.get("rf_team", "all")

    st.checkbox("Show Fallers",        key="rf_show_fallers")
    st.checkbox("Show min PA",         key="rf_show_min_pa")
    st.checkbox("Show min innings",    key="rf_show_min_inn")
    st.checkbox("Show player PA",      key="rf_show_player_pa")
    if not alltime_mode:
        st.checkbox("Show player innings", key="rf_show_innings")

show_fallers = st.session_state.get("rf_show_fallers", False)

if not alltime_mode and stat == "FRV" and start_year < 2018:
    st.warning("⚠️ FRV may be understated for catchers before 2018 due to missing framing data from Baseball Savant.")

# ----------------------------
#  LOAD DATA
# ----------------------------
if alltime_mode:
    alltime_df_full = load_alltime_csv()
    if alltime_df_full.empty:
        st.error("yoy_deltas.csv not found in app folder. Run precompute_yoy.py first.")
        st.stop()
    delta_col = f"{stat}_delta"
    start_col = f"{stat}_start"
    end_col   = f"{stat}_end"
    df = alltime_df_full[
        (pd.to_numeric(alltime_df_full["PA_start"], errors="coerce") >= ALLTIME_MIN_PA) &
        (pd.to_numeric(alltime_df_full["PA_end"],   errors="coerce") >= ALLTIME_MIN_PA)
    ].copy()
    df = df.rename(columns={
        "TeamDisplay_end": "TeamDisplay",
        delta_col:         stat,
        start_col:         f"{stat}_start",
        end_col:           f"{stat}_end",
    })
    df["PA_start"] = pd.to_numeric(df["PA_start"], errors="coerce")
    df["PA_end"]   = pd.to_numeric(df["PA_end"],   errors="coerce")
elif end_year > start_year:
    with st.spinner("Loading data..."):
        df = load_risers_data(start_year, end_year, min_pa_val, position_val, team_val)
else:
    df = pd.DataFrame()

# Apply minimum innings filter (uses TotalInn from fielding merge in load_single_year)
# We need innings for both years — store them in load_risers_data
if not df.empty and min_inn_val > 0 and "TotalInn_start" in df.columns and "TotalInn_end" in df.columns:
    inn_start = pd.to_numeric(df["TotalInn_start"], errors="coerce").fillna(0)
    inn_end   = pd.to_numeric(df["TotalInn_end"],   errors="coerce").fillna(0)
    df = df[(inn_start >= min_inn_val) & (inn_end >= min_inn_val)]
elif not df.empty and min_inn_val > 0 and "TotalInn" in df.columns:
    # fallback: only end year innings available
    inn_end = pd.to_numeric(df["TotalInn"], errors="coerce").fillna(0)
    df = df[inn_end >= min_inn_val]

if not df.empty and stat in FIELDING_STATS:
    player_names = df["Name"].tolist()
    f_start = load_fielding_for_year(player_names, start_year)
    f_end   = load_fielding_for_year(player_names, end_year)

    df["NameKey"] = df["Name"].apply(normalize_statcast_name)

    for fs, suffix in [(f_start, "_start"), (f_end, "_end")]:
        if fs.empty:
            continue
        for fcol in ["FRV", "ARM", "RANGE", "OAA", "DRS", "TZ", "UZR", "FRM"]:
            if fcol not in fs.columns:
                continue
            merged = df[["NameKey"]].merge(fs[["NameKey", fcol]], on="NameKey", how="left")
            df[f"{fcol}{suffix}"] = merged[fcol].values  # no fillna — keep NaN if not found

    for fcol in ["FRV", "ARM", "RANGE", "OAA", "DRS", "TZ", "UZR", "FRM"]:
        s_col = f"{fcol}_start"
        e_col = f"{fcol}_end"
        if s_col in df.columns and e_col in df.columns:
            s = pd.to_numeric(df[s_col], errors="coerce")
            e = pd.to_numeric(df[e_col], errors="coerce")
            # Only compute delta where both years have data
            df[fcol] = e - s
        elif e_col in df.columns:
            df[fcol] = pd.to_numeric(df[e_col], errors="coerce")

if not df.empty and stat in df.columns:
    stat_is_lower_better = stat in lower_better
    if stat_is_lower_better:
        ascending = not show_fallers
    else:
        ascending = show_fallers

    df = df.sort_values(by=stat, ascending=ascending)

    # Only show improvements for risers, only declines for fallers
    numeric_delta = pd.to_numeric(df[stat], errors="coerce")
    if stat_is_lower_better:
        df = df[numeric_delta < 0] if not show_fallers else df[numeric_delta > 0]
    else:
        df = df[numeric_delta > 0] if not show_fallers else df[numeric_delta < 0]

    df = df.head(10)
elif not df.empty:
    st.error(f"Column '{stat}' not found. Available columns: {', '.join(df.columns)}")
    df = pd.DataFrame()

cards = []
for _, row in df.iterrows():
    name    = row.get("Name", "")
    team    = row.get("TeamDisplay", "")
    raw_delta = row.get(stat, np.nan)
    display_delta = raw_delta

    transformed_delta = display_delta
    # Contact%/Whiff% delta is already on 0-100 scale after conversion in load_single_year
    # Divide by 100 so format_stat's '* 100 if v <= 1' brings it back to the right value
    if stat == "Contact%" and pd.notna(transformed_delta):
        transformed_delta = float(transformed_delta) / 100
    is_positive = pd.notna(transformed_delta) and float(transformed_delta) > 0
    display_val = format_stat(stat, transformed_delta, show_sign=is_positive)

    end_val_raw = row.get(f"{stat}_end", np.nan)
    if stat == "Contact%" and pd.notna(end_val_raw):
        end_val_raw = float(end_val_raw) / 100
    end_display = format_stat(stat, end_val_raw) if pd.notna(end_val_raw) else ""
    stat_label = label_map.get(stat, stat)
    team_subtitle = f"{team}"

    src_row = row
    try:
        pos = list(df.index).index(row.name)
        key = f"rf_mlbam_override_{pos}"
        override_val = st.session_state.get(key, "")
        if override_val and str(override_val).strip():
            try:
                src_row = row.copy()
                src_row["mlbam_override"] = int(str(override_val).strip())
            except Exception:
                pass
    except Exception:
        pass

    pa_start = row.get("PA_start", np.nan)
    pa_end   = row.get("PA_end",   np.nan)
    player_pa_display = ""
    if alltime_mode:
        # Show the year pair on each card
        yr_s = row.get("start_year", "")
        yr_e = row.get("end_year", "")
        if pd.notna(yr_s) and pd.notna(yr_e):
            player_pa_display = f'<div class="player-pa">{int(yr_s)} → {int(yr_e)}</div>'
    elif st.session_state.get("rf_show_player_pa", False):
        parts_pa = []
        if pd.notna(pa_start):
            parts_pa.append(f"{int(pa_start)}")
        if pd.notna(pa_end):
            parts_pa.append(f"{int(pa_end)}")
        if parts_pa:
            player_pa_display = f'<div class="player-pa">{" → ".join(parts_pa)} PA</div>'

    innings_val = row.get("TotalInn", np.nan)
    player_innings_display = (
        f'<div class="player-innings">{int(innings_val)} Inn.</div>'
        if st.session_state.get("rf_show_innings", False) and pd.notna(innings_val) and innings_val > 0
        else ""
    )

    src = get_headshot_url_from_row(src_row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(str(name))}"/>'

    if stat in lower_better:
        is_improvement = pd.notna(transformed_delta) and float(transformed_delta) < 0
    else:
        is_improvement = is_positive
    delta_class = "stat-positive" if is_improvement else "stat-negative"

    end_context = f'<div class="player-endval">{stat_label}: {end_display}</div>' if end_display else ""

    card_html = f'''
    <div class="player-card">
      {img_html}
      <div class="player-name">{name}</div>
      <div class="player-team">{team_subtitle}</div>
      <div class="player-stat {delta_class}">{display_val}</div>
      {end_context}
      {player_pa_display}
      {player_innings_display}
    </div>
    '''
    cards.append(card_html)

title_stat_label = label_map.get(stat, stat)
pos_suffix   = f" ({POSITION_OPTIONS.get(position_val, '')})" if position_val != "all" else ""
team_prefix  = f"{TEAM_OPTIONS.get(team_val, '')} " if team_val != "all" else ""
riser_label  = "Fallers" if show_fallers else "Risers"

if alltime_mode:
    title = f"All-Time {title_stat_label} {riser_label}{pos_suffix}"
else:
    title = f"Top {team_prefix}{title_stat_label} {riser_label}{pos_suffix}: {int(start_year)} → {int(end_year)}"

min_pa_subtitle = ""
if st.session_state.get("rf_show_min_pa", False):
    display_min_pa = ALLTIME_MIN_PA if alltime_mode else min_pa_val
    min_pa_subtitle = f'<div class="leaderboard-subtitle">Min {display_min_pa} PA each year</div>'
if st.session_state.get("rf_show_min_inn", False) and min_inn_val > 0:
    min_pa_subtitle += f'<div class="leaderboard-subtitle">Min {min_inn_val} Inn each year</div>'

footer_middle = ""
if position_val != "all" and stat in FIELDING_STATS:
    footer_middle = f'<p>Total {title_stat_label} among primary {POSITION_OPTIONS.get(position_val, position_val)}</p>'

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{title}</div>
    {min_pa_subtitle}
    <div class="players-grid">
        {''.join(cards) if cards else '<div style="padding:2rem;color:#999;">No data found. Try adjusting your filters or years.</div>'}
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
    padding: 2rem 2rem;
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
}}
.player-card {{
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
.player-name {{
    font-weight: 800;
    margin-top: 0.35rem;
    font-size: 1.1rem;
}}
.player-team {{
    color: #666;
    font-size: 0.85rem;
}}
.player-stat {{
    font-weight: 900;
    font-size: 1.5rem;
    margin-top: 0.25rem;
}}
.stat-positive {{
    color: #1a7a3c;
}}
.stat-negative {{
    color: #c0392b;
}}
.player-endval {{
    color: #888;
    font-size: .9rem;
    margin-top: 0.1rem;
}}
.player-pa {{
    color: #666;
    font-size: 0.9rem;
    margin-top: 0.1rem;
}}
.player-innings {{
    color: #666;
    font-size: 0.9rem;
}}
html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    width: 100%;
}}
.footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1.5rem;
}}
.footer p {{
    margin: 0;
    font-size: 0.9rem;
    color: #666;
    flex: 1;
    text-align: center;
}}
.footer p:first-child {{ text-align: left; }}
.footer p:last-child  {{ text-align: right; }}
</style>
</head>
<body>
{grid_html}
</body>
</html>
"""

with col2:
    components.html(full_html, height=850)

if not df.empty:
    st.markdown("---")
    st.write("Manual MLBAM overrides (enter MLBAM id to fix headshot)")

    for row_offset in range(0, 10, 5):
        cols_row = st.columns(5)
        for col_idx in range(5):
            player_idx = row_offset + col_idx
            if player_idx >= len(df):
                break
            idx = df.index[player_idx]
            row = df.loc[idx]
            with cols_row[col_idx]:
                key = f"rf_mlbam_override_{player_idx}"
                default_val = ""
                if "mlbam_override" in df.columns and pd.notna(row.get("mlbam_override")):
                    try:
                        default_val = str(int(row["mlbam_override"]))
                    except Exception:
                        default_val = str(row.get("mlbam_override", ""))
                user_val = st.text_input(f"Player {player_idx+1} MLBAM", value=default_val, key=key)
                try:
                    df.at[idx, "mlbam_override"] = int(str(user_val).strip()) if user_val and str(user_val).strip() else np.nan
                except Exception:
                    df.at[idx, "mlbam_override"] = np.nan
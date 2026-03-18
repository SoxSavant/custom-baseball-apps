import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import html
import requests
import re
from pybaseball.statcast_fielding import statcast_outs_above_average
import io
from pathlib import Path
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.set_page_config(page_title="Hitter Stat Filter Leaderboard", layout="wide", page_icon="⚾")

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
#  CONSTANTS
# ----------------------------

POSITION_FILTER_MAP = {
    "all": None, "C": ["C"], "1B": ["1B"], "2B": ["2B"], "3B": ["3B"],
    "SS": ["SS"], "LF": ["LF"], "CF": ["CF"], "RF": ["RF"],
    "OF": ["LF", "CF", "RF", "OF"], "DH": ["DH"],
}
POSITION_OPTIONS = {
    "all": "All Positions", "C": "C", "1B": "1B", "2B": "2B", "3B": "3B",
    "SS": "SS", "LF": "LF", "CF": "CF", "RF": "RF", "OF": "OF", "DH": "DH",
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

# Stats available for combo filters
COMBO_STATS = [
    "WAR", "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "AVG", "OBP", "SLG", "ISO", "BABIP",
    "HR", "SB", "RBI", "R", "H", "1B", "2B", "3B", "XBH", "TB", "BB", "IBB", "SO", "PA", "AB", "G",
    "K%", "BB%", "K-BB%", "O-Swing%", "Contact%",
    "Barrel%", "HardHit%", "EV",
    "GB%", "FB%", "LD%", "Pull%",
    "Age", "Off", "Def", "BsR", "WPA", "Clutch",
    "FRV", "OAA", "ARM", "DRS", "TZ", "UZR", "FRM",
]

FIELDING_STATS = {"FRV", "OAA", "ARM", "DRS", "TZ", "UZR", "FRM"}

# Stats where lower = better (default operator flips to <=)
LOWER_BETTER = {"K%", "O-Swing%", "Contact%", "SO", "GB%"}

label_map = {
    "HardHit%": "Hard Hit%", "WAR": "fWAR", "EV": "Avg Exit Velo",
    "Contact%": "Whiff%", "O-Swing%": "Chase%",
}

# Default thresholds per stat
STAT_DEFAULTS = {
    "HR": 30, "SB": 30, "RBI": 100, "R": 100, "H": 150,
    "WAR": 4.0, "wRC+": 130, "wOBA": 0.370, "OPS": 0.900,
    "xwOBA": 0.370, "xBA": 0.280, "xSLG": 0.480,
    "AVG": 0.300, "OBP": 0.370, "SLG": 0.500, "ISO": 0.200,
    "K%": 20.0, "BB%": 10.0, "Barrel%": 12.0, "HardHit%": 45.0,
    "EV": 92.0, "BB": 60, "IBB": 10, "SO": 100, "PA": 502, "AB": 450,
    "2B": 30, "1B": 100, "3B": 5, "XBH": 50, "TB": 250, "G": 140,
    "Age": 30, "Clutch": 1.0, "FRV": 10, "OAA": 10, "ARM": 3, "DRS": 10, "TZ": 5, "UZR": 5, "FRM": 10,
}

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
MAX_DISPLAY = 30

# ----------------------------
#  HELPERS
# ----------------------------

def update_stat_default(i):
    stat = st.session_state[f"sc_stat_{i}"]
    st.session_state[f"sc_val_{i}"] = float(STAT_DEFAULTS.get(stat, 0.0))
    st.session_state[f"sc_op_{i}"] = "<=" if stat in LOWER_BETTER else ">="

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


def normalize_team_code(team: str, year: int):
    if not team:
        return None
    team = team.upper().strip()
    if team in {"", "-", "--", "---", "TOT"}:
        return None
    if year < 2025:
        if team in {"ATH", "OAK"}:
            return "OAK"
    else:
        if team in {"ATH", "OAK"}:
            return "ATH"
    return team


def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper = stat.upper()
    if upper in {"FRV", "ARM"}:
        return f"{int(round(float(val)))}"
    if upper in {"WAR", "OFF", "DEF", "BSR", "EV", "AVG EXIT VELO"}:
        v = float(val)
        return f"{v:.1f}" if abs(v - round(v)) >= 1e-9 else f"{int(round(v))}.0"
    if upper in {"WPA"}:
        return f"{float(val):.2f}"
    if upper in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO"}:
        return f"{float(val):.3f}".lstrip("0") or ".000"
    if upper in {"WRC+"}:
        return f"{int(round(float(val)))}"
    if "%" in stat or any(x in stat for x in ["Barrel", "Hard", "K%", "Swing", "Whiff"]):
        v = float(val)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"
    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def format_threshold(stat: str, val: float, op: str) -> str:
    label = label_map.get(stat, stat)
    formatted = format_stat(stat, val)
    # Strip trailing % so "10%+ K%" becomes "10+ K%"
    numeric = formatted.rstrip("%")
    if op == ">=":
        return f"{numeric}+ {label}"
    else:
        return f"≤{numeric} {label}"


# ----------------------------
#  DATA LOADING
# ----------------------------

@st.cache_data(ttl=600, max_entries=10)
def batting_stats_cached(year: int, qual: int = 0):
    try:
        df = pybaseball.batting_stats(year, year, qual=qual, split_seasons=False)
        return df if df is not None and not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, max_entries=10)
def fielding_stats_cached(year: int):
    try:
        df = pybaseball.fielding_stats(year, year, qual=0)
        return df if df is not None and not df.empty else pd.DataFrame()
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
            estimated = (float(pa) / 4.1) * 9
            fi = fielding.loc[fielding["IDfg"] == fg_id, "TotalInn"].values if "TotalInn" in fielding.columns else []
            fi = float(fi[0]) if len(fi) > 0 else 0
            if fi == 0 or (estimated / fi) > 3:
                if fg_id in fielding["IDfg"].values:
                    fielding.loc[fielding["IDfg"] == fg_id, "DefPos"] = "DH"
                else:
                    fielding = pd.concat([fielding, pd.DataFrame([{"IDfg": fg_id, "DefPos": "DH", "TotalInn": 0}])], ignore_index=True)
    cols = ["IDfg", "DefPos", "TotalInn"] if "TotalInn" in fielding.columns else ["IDfg", "DefPos"]
    return fielding[cols]


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
    oaa_col = next((c for c in ["outs_above_average", "oaa"] if c in df.columns), None)
    if not oaa_col:
        return pd.DataFrame()
    df["OAA"] = pd.to_numeric(df[oaa_col], errors="coerce")
    return df[["NameKey", "Name", "OAA"]]


@st.cache_data(show_spinner=False, ttl=600, max_entries=5)
def load_fangraphs_fielding(player_names: list, year: int) -> pd.DataFrame:
    if not player_names:
        return pd.DataFrame()
    try:
        df = pybaseball.fielding_stats(year, year, qual=0)
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
def load_fielding_for_players(player_names: list, year: int) -> pd.DataFrame:
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
    fg_data = load_fangraphs_fielding(player_names, year)
    if not savant_data.empty and not fg_data.empty:
        return savant_data.merge(fg_data, on="NameKey", how="outer")
    elif not savant_data.empty:
        return savant_data
    elif not fg_data.empty:
        return fg_data
    return pd.DataFrame()


@st.cache_data(ttl=600, max_entries=10)
def load_year(year: int, min_pa: int = 0, position: str = "all") -> pd.DataFrame:
    df = batting_stats_cached(year, qual=min_pa)
    if df is None or df.empty:
        return pd.DataFrame()

    df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")

    # Resolve TOT rows
    tot_ids = set(df.loc[df["Team"] == "TOT", "IDfg"])
    has_ind = set(df.loc[df["Team"] != "TOT", "IDfg"])
    non_tot = df[df["Team"] != "TOT"]
    tot_fb  = df[(df["Team"] == "TOT") & (df["IDfg"].isin(tot_ids - has_ind))]
    df = pd.concat([non_tot, tot_fb], ignore_index=True)
    df = df[df["Team"].notna()]

    fielding = get_primary_fielding(year, batting_df=df)
    if not fielding.empty:
        df = df.merge(fielding, on="IDfg", how="left")

    def is_junk_team(t: str) -> bool:
        # True for TOT, blanks, and any dash-only strings like "---" or "- - -"
        return t == "TOT" or t.replace(" ", "").replace("-", "") == ""

    # Collapse multi-team players — always show "2+ Teams" if traded
    collapsed = []
    for fg_id, grp in df.groupby("IDfg"):
        raw_teams = grp["Team"].dropna().astype(str).str.strip().str.upper().tolist()
        teams = sorted(set(
            t for t in [normalize_team_code(x, year) for x in raw_teams if not is_junk_team(x)]
            if t
        ))
        tot_row = grp[grp["Team"] == "TOT"]
        base = tot_row.iloc[0].to_dict() if not tot_row.empty else grp.iloc[0].to_dict()
        if len(teams) == 1:
            base["TeamDisplay"] = teams[0]
        elif len(teams) > 1:
            base["TeamDisplay"] = "2+ Teams"
        else:
            fallback = next((x for x in raw_teams if not is_junk_team(x)), "")
            base["TeamDisplay"] = fallback if fallback else "2+ Teams"
        base["_teams_list"] = teams
        if "DefPos" in grp.columns:
            nd = grp.dropna(subset=["DefPos"])
            base["DefPos"] = nd["DefPos"].iloc[0] if not nd.empty else base.get("DefPos", "")
        collapsed.append(base)

    df = pd.DataFrame(collapsed)

    # Derived stats
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

    # Position filter
    if position != "all" and "DefPos" in df.columns:
        pos_values = POSITION_FILTER_MAP.get(position, [])
        df["DefPos"] = df["DefPos"].astype(str).str.upper()
        df = df[df["DefPos"].isin([p.upper() for p in pos_values])]

    return df


# ----------------------------
#  HEADSHOT FUNCTIONS
# ----------------------------

@st.cache_data(show_spinner=False)
def lookup_mlbam_id(full_name: str, return_bbref: bool = False):
    if not full_name or not full_name.strip():
        return (None, None) if return_bbref else None
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

    def normalize_token(tok):
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

    parts = full_name.split()
    while parts and parts[-1].lower() in suffixes:
        parts.pop()
    if len(parts) < 2:
        return (None, None) if return_bbref else None

    first_raw = parts[0]
    last_raw = " ".join(parts[1:])
    target_clean = clean_full(first_raw + last_raw)

    variants = [
        (last_raw, first_raw),
        (normalize_token(last_raw), normalize_token(first_raw)),
        (normalize_token(last_raw).lower(), normalize_token(first_raw).lower()),
        (last_raw.replace(".", ""), first_raw.replace(".", "")),
    ]

    best_mlbam = None; best_bbref = None; first_mlbam = None; first_bbref = None

    def consider(row):
        nonlocal best_mlbam, best_bbref, first_mlbam, first_bbref
        combo = clean_full(str(row.get("name_first","")) + str(row.get("name_last","")))
        mv = row.get("key_mlbam"); bv = row.get("key_bbref")
        if combo == target_clean:
            if pd.notna(mv):
                try: best_mlbam = int(mv)
                except: pass
            if pd.notna(bv):
                try: best_bbref = str(bv)
                except: pass
        if first_mlbam is None and pd.notna(mv):
            try: first_mlbam = int(mv)
            except: pass
        if first_bbref is None and pd.notna(bv):
            try: first_bbref = str(bv)
            except: pass

    for last, first in variants:
        try:
            ldf = pybaseball.playerid_lookup(last, first)
            if ldf is not None and not ldf.empty:
                for _, r in ldf.iterrows(): consider(r)
        except: continue

    mlbam_res = best_mlbam if best_mlbam is not None else first_mlbam
    bbref_res = best_bbref if best_bbref is not None else first_bbref
    return (mlbam_res, bbref_res) if return_bbref else mlbam_res


@st.cache_data(show_spinner=False, ttl=21600)
def build_mlb_headshot(mlbam) -> str | None:
    if mlbam is None: return None
    mlbam_val = str(mlbam).strip()
    headers = {"User-Agent": HEADSHOT_USER_AGENT}
    for base in HEADSHOT_BASES:
        try:
            url = base.format(mlbam=mlbam_val)
            resp = requests.head(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200: return url
            if resp.status_code in (403, 404, 405):
                r2 = requests.get(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, stream=True)
                if r2.status_code == 200: return url
        except: continue
    return HEADSHOT_BASES[0].format(mlbam=mlbam_val)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def reverse_lookup_mlbam(fg_id: int) -> int | None:
    try:
        rev = pybaseball.playerid_reverse_lookup([int(fg_id)], key_type="fangraphs")
        if rev is not None and not rev.empty:
            v = rev.iloc[0].get("key_mlbam")
            if pd.notna(v): return int(v)
    except: pass
    return None


def get_headshot_url_from_row(row: pd.Series) -> str:
    name = str(row.get("Name", "")).strip()
    for col in ["mlbam_override","mlbamid","mlbam_id","mlbam","MLBID","MLBAMID","key_mlbam"]:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                try:
                    h = build_mlb_headshot(int(val))
                    if h: return h
                except: pass
    for col in ["playerid","IDfg","fg_id","FGID"]:
        if col in row.index:
            fg = row.get(col)
            if pd.notna(fg) and str(fg).strip():
                try:
                    mlbam = reverse_lookup_mlbam(int(fg))
                    if mlbam:
                        h = build_mlb_headshot(mlbam)
                        if h: return h
                except: pass
    if name:
        mlbam_fb, _ = lookup_mlbam_id(name, return_bbref=True)
        if mlbam_fb:
            h = build_mlb_headshot(mlbam_fb)
            if h: return h
    return HEADSHOT_PLACEHOLDER


# ----------------------------
#  PAGE HEADER
# ----------------------------
title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Hitter Stat Filter Leaderboard")
with meta_col:
    st.markdown(
        '<div style="text-align:right;font-size:1rem;padding-top:0.6rem;">'
        'Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a></div>',
        unsafe_allow_html=True,
    )

current_year = date.today().year

# ----------------------------
#  SESSION STATE
# ----------------------------
for key, default in [
    ("sc_year",       2025),
    ("sc_min_pa",     300),
    ("sc_position",   "all"),
    ("sc_team",       "all"),

    ("sc_stat_0",     "HR"),   ("sc_op_0",  ">="), ("sc_val_0",  25.0),
    ("sc_stat_1",     "SB"),   ("sc_op_1",  ">="), ("sc_val_1",  30.0),
    ("sc_show_pa",    False),
    ("sc_top10", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
    <style>
        .stSelectbox div[data-baseweb="select"],
        .stNumberInput > div { max-width: 200px; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
#  CONTROLS
# ----------------------------
col1, col2 = st.columns([0.5, 2])

with col1:
    num_stats = st.radio("Number of stat filters", [1, 2, 3, 4], index=1, horizontal=True, key="sc_num_stats")
    st.number_input("Year", min_value=1900, max_value=current_year, key="sc_year")
    st.number_input("Min PA", min_value=0, max_value=20000, key="sc_min_pa")

    # Per-stat filter rows
    for i in range(num_stats):
        st.markdown(f"**Stat {i+1}**")
        stat_key = f"sc_stat_{i}"
        op_key   = f"sc_op_{i}"
        val_key  = f"sc_val_{i}"

        current_stat = st.session_state.get(stat_key)
        stat_index = COMBO_STATS.index(current_stat) if current_stat in COMBO_STATS else 0

        new_stat = st.selectbox(
            f"Stat {i+1}",
            COMBO_STATS,
            index=stat_index,
            key=stat_key,
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed",
            on_change=update_stat_default,
            args=(i,),
        )

        default_op = "<=" if new_stat in LOWER_BETTER else ">="
        current_op = st.session_state.get(op_key, default_op)
        op_index = 0 if current_op == ">=" else 1

        op_col, val_col = st.columns([1, 2])
        with op_col:
            op = st.selectbox("Op", [">=", "<="],
                              index=op_index,
                              key=op_key, label_visibility="collapsed")
        with val_col:
            default_val = STAT_DEFAULTS.get(new_stat, 0.0)
            current_val = st.session_state.get(val_key, default_val)
            # Three tiers: rate stats need 3dp, % stats need 1dp, counting stats are integers
            RATE_STATS_3DP = {"AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "ISO", "BABIP"}
            if new_stat in RATE_STATS_3DP:
                step = 0.001
                fmt = "%.3f"
            elif "%" in new_stat or new_stat in {"EV"}:
                step = 0.1
                fmt = "%.1f"
            else:
                step = 1.0
                fmt = "%.0f"
            st.number_input(
                f"Value {i+1}", step=step,
                key=val_key, label_visibility="collapsed",
                format=fmt,
            )
    st.selectbox("Position", options=list(POSITION_OPTIONS.keys()),
                 format_func=lambda x: POSITION_OPTIONS[x], key="sc_position")
    st.selectbox("Team", options=list(TEAM_OPTIONS.keys()),
                 format_func=lambda x: TEAM_OPTIONS[x], key="sc_team")

    st.checkbox("Show player PA", key="sc_show_pa")
    st.checkbox("Only display top 10", key="sc_top10")

# ----------------------------
#  LOAD & FILTER DATA
# ----------------------------
year_val     = int(st.session_state["sc_year"])
min_pa_val   = int(st.session_state["sc_min_pa"])
position_val = st.session_state["sc_position"]
team_val     = st.session_state["sc_team"]

with st.spinner("Loading data..."):
    df = load_year(year_val, min_pa=min_pa_val, position=position_val)

# Team filter
if team_val != "all" and not df.empty and "_teams_list" in df.columns:
    df = df[df["_teams_list"].apply(
        lambda tl: any(normalize_team_code(t, year_val) == team_val for t in (tl or []))
    )]

# Build active filters
active_filters = []
for i in range(num_stats):
    stat = st.session_state.get(f"sc_stat_{i}")
    op   = st.session_state.get(f"sc_op_{i}", ">=")
    val  = st.session_state.get(f"sc_val_{i}", 0.0)
    if stat:
        active_filters.append((stat, op, float(val)))

# Merge fielding stats if any filter uses them
if not df.empty:
    fielding_needed = [s for s, _, _ in active_filters if s in FIELDING_STATS]
    if fielding_needed:
        player_names = df["Name"].tolist()
        fielding_data = load_fielding_for_players(player_names, year_val)
        if not fielding_data.empty:
            df["NameKey"] = df["Name"].apply(normalize_statcast_name)
            df = df.merge(fielding_data, on="NameKey", how="left", suffixes=("", "_fielding"))
        for fcol in FIELDING_STATS:
            if fcol in df.columns:
                df[fcol] = pd.to_numeric(df[fcol], errors="coerce").fillna(0)

# Apply each filter
PCT_STATS = {
    "K%", "BB%", "K-BB%", "O-Swing%", "Contact%",
    "Barrel%", "HardHit%", "GB%", "FB%", "LD%", "Pull%",
}
total_qualified = 0
if not df.empty:
    mask = pd.Series([True] * len(df), index=df.index)
    for stat, op, val in active_filters:
        if stat not in df.columns:
            continue
        col_vals = pd.to_numeric(df[stat], errors="coerce")
        compare_val = val
        if stat in PCT_STATS:
            median_col = col_vals.median()
            if pd.notna(median_col) and median_col <= 1:
                if val > 1:
                    compare_val = val / 100
            if stat == "Contact%":
                contact_threshold = 1 - (compare_val if compare_val <= 1 else compare_val / 100)
                if op == ">=":
                    mask = mask & (col_vals <= contact_threshold)
                else:
                    mask = mask & (col_vals >= contact_threshold)
                continue
        if op == ">=":
            mask = mask & (col_vals >= compare_val)
        else:
            mask = mask & (col_vals <= compare_val)
    df = df[mask]
    total_qualified = len(df)
    if active_filters:
        sort_stat, sort_op, _ = active_filters[0]
        if sort_stat in df.columns:
            asc = (sort_stat in LOWER_BETTER and sort_op == "<=")
            df = df.sort_values(sort_stat, ascending=asc)
    display_limit = 10 if st.session_state.get("sc_top10") and total_qualified > 10 else MAX_DISPLAY
    if total_qualified > display_limit:
        df = df.head(display_limit)

# ----------------------------
#  BUILD CARDS
# ----------------------------
cards = []
for card_pos, (_, row) in enumerate(df.iterrows()):
    name = row.get("Name", "")
    team = row.get("TeamDisplay", "")

    stat_lines = []
    for stat, op, threshold in active_filters:
        val = row.get(stat, np.nan)
        if pd.notna(val):
            display = format_stat(stat, val)
            lbl = label_map.get(stat, stat)
            stat_lines.append(f'<span class="stat-label">{lbl}:</span> <span class="stat-value">{display}</span>')

    pa_val = row.get("PA", np.nan)
    pa_display = f'<div class="player-pa">{int(pa_val)} PA</div>' if st.session_state.get("sc_show_pa") and pd.notna(pa_val) else ""

    src_row = row
    try:
        ov = st.session_state.get(f"sc_mlbam_override_{card_pos}", "")
        if ov and str(ov).strip():
            src_row = row.copy()
            src_row["mlbam_override"] = int(str(ov).strip())
    except: pass

    src = get_headshot_url_from_row(src_row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(str(name))}" width="155" height="155" style="object-fit:cover;border-radius:6px;border:1px solid #e0e0e0;background:#f6f6f6;display:block;"/>'

    cards.append(f'''
    <div class="player-card">
      {img_html}
      <div class="player-name">{html.escape(str(name))}</div>
      <div class="player-team">{html.escape(str(team))}</div>
      {'<div class="player-stat-line">' + " | ".join(stat_lines) + "</div>" if stat_lines else ""}
      {pa_display}
    </div>''')

# ----------------------------
#  TITLE
# ----------------------------
filter_parts = [format_threshold(s, v, op) for s, op, v in active_filters]
filter_str = ", ".join(filter_parts)
pos_suffix  = f" ({POSITION_OPTIONS[position_val]})" if position_val != "all" else ""
team_suffix = f"({team_val}) " if team_val != "all" else ""
title = f"{filter_str} in {year_val} {team_suffix}{pos_suffix}"

overflow_note = ""
display_limit = 10 if st.session_state.get("sc_top10") and total_qualified > 10 else MAX_DISPLAY
if total_qualified > display_limit:
    overflow_note = f'<div class="overflow-note">Showing top {display_limit} of {total_qualified} qualifying players</div>'

if not cards:
    body = '<div style="padding:2rem;color:#999;text-align:center;">No players matched all filters. Try adjusting your thresholds.</div>'
else:
    body = "".join(cards)

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{html.escape(title)}</div>
    {overflow_note}
    <div class="players-grid">{body}</div>
    <div class="footer">
        <p>By: Sox_Savant</p>
        <p>Data: FanGraphs</p>
    </div>
</div>
"""

card_count = len(cards)
est_rows = max(1, (card_count + 4) // 5)
est_height = 120 + est_rows * 280 + 80

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style>
* 
html, body {{ background: transparent; font-family: "Source Sans Pro", sans-serif; }}
.leaderboard-card {{
    background: #fff;
    border: 1px solid #d0d0d0;
    border-radius: 12px;
    padding: 3rem 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto;
    width: 100%;
    max-width: 900px;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.25rem;
    margin-bottom: 1.2rem;
    text-align: center;
    line-height: 1.2;
}}
.overflow-note {{
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 1rem;
    margin-top: -0.5rem;
}}
.players-grid {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2rem 1rem;
}}
.player-card {{
    flex: 0 0 155px;
    width: 155px;
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
    font-size: 1rem;
    margin-top: 0.35rem;
    line-height: 1.2;
}}
.player-team {{
    color: #666;
    font-size: 0.8rem;
    margin-bottom: 0.25rem;
}}
.player-stat-line {{
    text-align: center;
    font-size: 0.95rem;
    margin-top: 0.15rem;
}}
.stat-label {{ color: #888; font-size: 0.85rem; }}
.stat-value {{ font-weight: 800; font-size: 0.95rem; color: #1a1a1a; }}
.player-pa {{ color: #aaa; font-size: 0.8rem; margin-top: 0.1rem; }}
.footer {{
    display: flex;
    justify-content: space-between;
    margin-top: 1.5rem;
    padding: 0 4rem;
}}
.footer p {{ margin: 0; font-size: 1rem; color: #888; flex: 1; text-align: center; }}
.footer p:first-child {{ text-align: left; }}
.footer p:last-child {{ text-align: right; }}
</style>
</head>
<body>
{grid_html}
</body>
</html>"""

with col2:
    components.html(full_html, height=est_height, scrolling=True)

# ----------------------------
#  MLBAM OVERRIDES
# ----------------------------
if not df.empty:
    st.markdown("---")
    st.write("Manual MLBAM overrides (enter MLBAM id to fix headshot)")
    n = len(df)
    for row_offset in range(0, min(n, MAX_DISPLAY), 5):
        cols_row = st.columns(5)
        for col_idx in range(5):
            player_idx = row_offset + col_idx
            if player_idx >= n: break
            idx = df.index[player_idx]
            row = df.loc[idx]
            with cols_row[col_idx]:
                key = f"sc_mlbam_override_{player_idx}"
                default_val = ""
                if "mlbam_override" in df.columns and pd.notna(row.get("mlbam_override")):
                    try: default_val = str(int(row["mlbam_override"]))
                    except: pass
                user_val = st.text_input(f"Player {player_idx+1} MLBAM", value=default_val, key=key)
                try:
                    df.at[idx, "mlbam_override"] = int(str(user_val).strip()) if user_val and str(user_val).strip() else np.nan
                except:
                    df.at[idx, "mlbam_override"] = np.nan

import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import html
import requests
from bs4 import BeautifulSoup
from pybaseball.statcast_fielding import statcast_outs_above_average
import io
from pathlib import Path
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.set_page_config(layout="wide")

# ----------------------------
#  MEMORY-OPTIMIZED DATA LOADING
# ----------------------------

@st.cache_data(ttl=3600, max_entries=3)  # Limit cache entries
def load_filtered_data(start_year, end_year, min_pa=0):
    """
    Load data and filter by PA threshold.
    For single year: use qual parameter (much faster!)
    For multi-year: filter AFTER aggregation (so min_pa represents total PA across all years).
    """
    if start_year == end_year:
        # Single year: use qual parameter for fast server-side filtering
        df = batting_stats(start_year, end_year, qual=min_pa, split_seasons=False)
        
        # FIX: Create proper TeamDisplay for single-year data
        # FanGraphs returns "---" or "- - -" for players who played for multiple teams
        if not df.empty and "Team" in df.columns:
            def make_team_display(team_val):
                if pd.isna(team_val):
                    return "N/A"
                team_str = str(team_val).strip()
                # FanGraphs uses these patterns for multi-team players
                if team_str in {"---", "- - -", "--", "TOT", ""}:
                    return "2 Teams"
                # Otherwise normalize the team code
                normalized = normalize_team_code(team_str, start_year)
                return normalized if normalized else "N/A"
            
            df["TeamDisplay"] = df["Team"].apply(make_team_display)
        
        return df
    else:
        # Multi-year: use smart pre-filter, then aggregate, then filter by total PA
        # Pre-filter estimate: if someone needs min_pa total, they likely averaged min_pa/num_years per season
        # Use a conservative estimate (divide by 2) to avoid filtering out players who had uneven distributions
        num_years = end_year - start_year + 1
        pre_filter_pa = max(1, min_pa // (num_years * 2)) if min_pa > 0 else 0
        
        frames = []
        for year in range(start_year, end_year + 1):
            # Use qual parameter to pre-filter on the server side
            yr_data = batting_stats(year, year, qual=pre_filter_pa, split_seasons=False)
            if not yr_data.empty:
                yr_data['Season'] = year
                frames.append(yr_data)
        
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            # Optimize dtypes BEFORE grouping
            combined = optimize_dtypes(combined)
            
            # Group by player and aggregate
            grouped_rows = []
            for player_id, grp in combined.groupby("IDfg"):
                name = grp["Name"].iloc[0] if not grp.empty else None
                row = aggregate_player_group(grp, name)
                if row is not None and len(row):
                    grouped_rows.append(row)
            
            result = pd.DataFrame(grouped_rows)
            # Optimize dtypes again after aggregation
            result = optimize_dtypes(result)
            
            # NOW filter by total PA across all years (final precise filter)
            if not result.empty and min_pa > 0:
                result = result[pd.to_numeric(result.get("PA", 0), errors="coerce").fillna(0) >= min_pa]
            
            return result
        return pd.DataFrame()


def optimize_dtypes(df):
    """Convert data types to use less memory"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Convert float64 to float32 where appropriate
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        # Keep high precision for rate stats, reduce for counting stats
        if col not in ['AVG', 'OBP', 'SLG', 'wOBA', 'xwOBA', 'xBA', 'xSLG']:
            df[col] = df[col].astype('float32')
    
    # Convert int64 to int32 where appropriate
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() < 2147483647:  # int32 max
            df[col] = df[col].astype('int32')
    
    return df


# ----------------------------
#  Helpers (unchanged from original)
# ----------------------------

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
        return "N/A"
    if len(teams) == 1:
        return teams[0]
    return f"{len(teams)} Teams"


# ----------------------------
#  External Data Loaders (SIMPLIFIED)
# ----------------------------

@st.cache_data(ttl=600, max_entries=2)  # Reduced TTL and entries
def batting_stats(start_year: int, end_year: int, qual=0, split_seasons=False):
    try:
        return pybaseball.batting_stats(start_year, end_year, qual=qual, split_seasons=split_seasons)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600, max_entries=10)
def load_savant_frv_year(year: int) -> pd.DataFrame:
    """Load Fielding Run Value from Baseball Savant for a specific year."""
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
    df = df.rename(
        columns={
            "name": "NameRaw",
            "total_runs": "FRV",
            "arm_runs": "ARM",
            "range_runs": "RANGE",
        }
    )
    df["Name"] = df["NameRaw"].astype(str).str.strip()
    df["NameKey"] = df["Name"].apply(normalize_statcast_name)
    for metric in ["FRV", "ARM", "RANGE"]:
        df[metric] = pd.to_numeric(df.get(metric), errors="coerce")
    return df[["NameKey", "Name", "FRV", "ARM", "RANGE"]]


@st.cache_data(show_spinner=False, ttl=600, max_entries=10)
def load_savant_oaa_year(year: int) -> pd.DataFrame:
    """Load Outs Above Average from Statcast for a specific year."""
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
    """
    Load DRS and TZ from FanGraphs fielding stats for specific players.
    """
    if not player_names:
        return pd.DataFrame()
    
    try:
        # Load FanGraphs fielding data for the year range
        df = pybaseball.fielding_stats(start_year, end_year, qual=0, split_seasons=False)
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Normalize names for matching
        df["NameKey"] = df["Name"].apply(normalize_statcast_name)
        target_keys = set([normalize_statcast_name(n) for n in player_names])
        
        # Filter to only our target players
        df = df[df["NameKey"].isin(target_keys)]
        
        if df.empty:
            return pd.DataFrame()
        
        # Aggregate by player (sum DRS, TZ, UZR, FRM across positions/years)
        result = df.groupby("NameKey", as_index=False).agg({
            "DRS": "sum",
            "TZ": "sum",
            "UZR": "sum",
            "FRM": "sum",
        })
        
        return result
        
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False, max_entries=5)
def load_fielding_for_players(player_names: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    """
    Load fielding stats ONLY for specific players (the top 10).
    Combines Savant data (FRV, OAA, ARM) with FanGraphs data (DRS, TZ, UZR, FRM).
    """
    if not player_names:
        return pd.DataFrame()
    
    # Normalize names for matching
    target_keys = set([normalize_statcast_name(n) for n in player_names])
    
    # Load Savant data (FRV, OAA, ARM)
    frames = []
    for year in range(start_year, end_year + 1):
        # Load FRV
        frv = load_savant_frv_year(year)
        if frv is not None and not frv.empty:
            frv = frv[frv["NameKey"].isin(target_keys)]
            if not frv.empty:
                frv["Season"] = year
                frames.append(frv)
        
        # Load OAA
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
        
        # Aggregate by player
        savant_data = combined.groupby("NameKey", as_index=False).agg({
            "FRV": "sum",
            "ARM": "sum",
            "RANGE": "sum",
            "OAA": "sum",
        })
    
    # Load FanGraphs data (DRS, TZ, UZR, FRM)
    fangraphs_data = load_fangraphs_fielding(player_names, start_year, end_year)
    
    # Merge the two datasets
    if not savant_data.empty and not fangraphs_data.empty:
        result = savant_data.merge(fangraphs_data, on="NameKey", how="outer")
    elif not savant_data.empty:
        result = savant_data
    elif not fangraphs_data.empty:
        result = fangraphs_data
    else:
        result = pd.DataFrame()
    
    return result

# Lightweight headshot helpers
HEADSHOT_BASES = [
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current",
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current",
]
HEADSHOT_PLACEHOLDER = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0nMjQwJyBoZWlnaHQ9JzI0MCcgdmlld0JveD0nMCAwIDI0MCAyNDAnIHhtbG5zPSdodHRwOi8v"
    "d3d3LnczLm9yZy8yMDAwL3N2Zyc+CjxyZWN0IHdpZHRoPScyNDAnIGhlaWdodD0nMjQwJyBmaWxsPScjZWVmJy8+"
    "CjxjaXJjbGUgY3g9JzEyMCcgY3k9Jzk1JyByPSc1NScgZmlsbD0nI2RkZScvPgo8Y2lyY2xlIGN4PScxMjAnIGN5"
    "PSc4NScgcj0nNDInIGZpbGw9JyNmZmYnIHN0cm9rZT0nI2NjYycvPgo8cGF0aCBkPSdNMTIwIDE1MGMtMzAgMC01"
    "NSAyNS01NSA1NXMzNSAxNS41IDU1IDE1LjUgNTUtMTUuNSA1NS0xNS41LTM1LTU1LTU1LTU1eicgZmlsbD0nI2Nj"
    "YycvPgo8L3N2Zz4="
)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)  # Limit cache size
def lookup_mlbam_id(full_name: str, return_bbref: bool = False):
    try:
        if not full_name or not isinstance(full_name, str):
            return (None, None) if return_bbref else None
        df = pybaseball.playerid_lookup(full_name)
        if df is None or df.empty:
            return (None, None) if return_bbref else None
        cols = [c for c in ("key_mlbam", "mlbam", "MLBAMID") if c in df.columns]
        mlbam = None
        if cols:
            val = df.iloc[0].get(cols[0])
            if pd.notna(val) and str(val).strip():
                try:
                    mlbam = int(val)
                except Exception:
                    mlbam = str(val).strip()
        bbref = None
        if return_bbref:
            if "key_bbref" in df.columns:
                v = df.iloc[0].get("key_bbref")
                if pd.notna(v) and str(v).strip():
                    bbref = str(v).strip()
        return (mlbam, bbref) if return_bbref else mlbam
    except Exception:
        return (None, None) if return_bbref else None


@st.cache_data(show_spinner=False, ttl=3600)
def build_mlb_headshot(mlbam: int | str | None) -> str | None:
    if mlbam is None:
        return None
    try:
        for base in HEADSHOT_BASES:
            try:
                return base.format(mlbam=int(mlbam))
            except Exception:
                try:
                    return base.format(mlbam=str(mlbam).strip())
                except Exception:
                    continue
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def reverse_lookup_mlbam(fg_id: int) -> int | None:
    """Reverse lookup mlbam from FanGraphs ID with caching."""
    try:
        rev = pybaseball.playerid_reverse_lookup([int(fg_id)], key_type="fangraphs")
        if rev is not None and not rev.empty:
            mlbam = rev.iloc[0].get("key_mlbam")
            if pd.notna(mlbam):
                return int(mlbam)
    except Exception:
        pass
    return None


def get_headshot_url_from_row(row: pd.Series) -> str:
    """Resolve the best headshot URL for a single-row (aggregated or yearly).
    
    Strategy:
    1. Look for mlbam-like columns and build an MLB headshot URL.
    2. If missing, look for FanGraphs id and reverse-lookup to mlbam.
    3. Return a placeholder when nothing resolved.
    """
    # 1) direct mlbam-like columns
    for col in ("mlbam", "MLBID", "key_mlbam", "mlbam_id", "MLBAMID", "mlbam_override"):
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                url = build_mlb_headshot(val)
                if url:
                    return url
    
    # 2) FanGraphs id reverse lookup (cached)
    for fg_col in ("playerid", "IDfg", "fg_id", "FGID"):
        if fg_col in row.index:
            fg = row.get(fg_col)
            if pd.notna(fg) and str(fg).strip():
                try:
                    mlbam = reverse_lookup_mlbam(int(fg))
                    if mlbam:
                        url = build_mlb_headshot(mlbam)
                        if url:
                            return url
                except Exception:
                    pass
    
    # 3) Fallback to placeholder
    return HEADSHOT_PLACEHOLDER


# ----------------------------
#  Aggregation (unchanged)
# ----------------------------

def aggregate_player_group(grp: pd.DataFrame, name: str | None = None) -> dict:
    result: dict[str, object] = {}

    if name is None and "Name" in grp.columns:
        val = grp["Name"].dropna()
        if not val.empty:
            name = str(val.iloc[0])
    if name:
        result["Name"] = name

    teams = grp.get("Team", pd.Series([], dtype=str)).dropna().astype(str).tolist()
    teams = [t.strip().upper() for t in teams if t.strip()]
    teams = [normalize_team_code(t, int(grp["Season"].iloc[0]) if "Season" in grp.columns else 2025) for t in teams]
    teams = collapse_athletics(sorted(set([t for t in teams if t])))

    result["Teams"] = teams
    result["TeamDisplay"] = compute_team_display(teams)

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
#  UI
# ----------------------------

STAT_ALLOWLIST = [
    "Off", "Def", "BsR", "WAR", "Barrel%", "HardHit%", "EV",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "H", "1B", "2B", "3B", "SB", "BB", "IBB", "SO",
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
}

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
if "hl_start_year" not in st.session_state:
    st.session_state["hl_start_year"] = 2025
if "hl_end_year" not in st.session_state:
    st.session_state["hl_end_year"] = 2025
if "hl_stat" not in st.session_state:
    st.session_state["hl_stat"] = "WAR"
if "hl_min_pa" not in st.session_state:
    st.session_state["hl_min_pa"] = 502

def on_year_change():
    s = st.session_state
    if not s.get("hl_span", False):
        s["hl_end_year"] = s["hl_start_year"]

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

stat = st.selectbox(
    "Stat",
    STAT_ALLOWLIST,
    key="hl_stat",
    format_func=lambda x: label_map.get(x, x),
)

if "hl_span" not in st.session_state:
    st.session_state["hl_span"] = False

col1, col2 = st.columns([.5, 2])

with col1:
    st.checkbox("Multi-year span", key="hl_span", on_change=on_year_change)
    start_label = "Start Year" if st.session_state.get("hl_span", False) else "Year"
    start_year = st.number_input(
        start_label,
        min_value=1900,
        max_value=current_year,
        key="hl_start_year",
        on_change=on_year_change
    )
    if st.session_state.get("hl_span", False):
        end_year = st.number_input(
            "End Year",
            min_value=2025,
            max_value=current_year,
            key="hl_end_year",
            on_change=on_year_change
        )
    else:
        end_year = st.session_state["hl_start_year"]
   
    min_pa = st.number_input(
        "Min PA",
        min_value=0,
        max_value=20000,
        key="hl_min_pa"
    )
    st.checkbox("Show worst", key="hl_sort_worst")
    st.checkbox("Show min PA", key="hl_show_min_pa")

# Load filtered data - min_pa applies to total PA across the selected span
min_pa_val = int(st.session_state.get("hl_min_pa", 0))
df = load_filtered_data(start_year, end_year, min_pa_val)

# FIX: If sorting by a fielding stat, we need to load fielding data for ALL qualifying players first
# Then we can sort and limit to top 10
if not df.empty and stat in ["FRV", "OAA", "ARM", "DRS", "TZ", "UZR", "FRM"]:
    # Get all player names that meet PA threshold
    player_names = df["Name"].tolist()
    fielding_data = load_fielding_for_players(player_names, start_year, end_year)
    
    if not fielding_data.empty:
        # Add NameKey to df for joining
        df["NameKey"] = df["Name"].apply(normalize_statcast_name)
        
        # Merge fielding stats
        df = df.merge(
            fielding_data,
            on="NameKey",
            how="left",
            suffixes=("", "_fielding")
        )
        
        # Fill NaN values with 0 for fielding stats so they can be sorted
        for col in ["FRV", "OAA", "ARM", "RANGE", "DRS", "TZ", "UZR", "FRM"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

# FIXED: Sort and LIMIT TO TOP 10
if stat in df.columns:
    df = df.sort_values(by=stat, ascending=st.session_state.get("hl_sort_worst", False))
    # *** CRITICAL FIX: LIMIT TO TOP 10 AFTER SORTING ***
    df = df.head(10)
else:
    st.error(f"Column '{stat}' not found. Available columns: {', '.join(df.columns)}")
    df = pd.DataFrame()  # Empty df if stat not found

# TeamDisplay should already be set by load_filtered_data or aggregate_player_group
# But ensure it exists as a safety measure
if not df.empty and "TeamDisplay" not in df.columns:
    df["TeamDisplay"] = "N/A"

cards = []
for _, row in df.iterrows():
    name = row.get("Name", "")
    team = row.get("TeamDisplay", "")
    raw_val = row.get(stat, np.nan)
    transformed = transform_stat_value(stat, raw_val)
    display_val = format_stat(stat, transformed)
    
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

    src = get_headshot_url_from_row(src_row) or ""
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(str(name))}"/>'
    card_html = f'''
    <div class="player-card">
      {img_html}
      <div class="player-name">{name}</div>
      <div class="player-team">{team}</div>
      <div class="player-stat">{display_val}</div>
    </div>
    '''
    cards.append(card_html)

span_label = f"{int(start_year)}" if start_year == end_year else f"{int(start_year)}–{int(end_year)}"
title_label = label_map.get(stat, stat)
title = f"{span_label} {title_label} Leaders"
if st.session_state.get("hl_sort_worst", False):
    title += " (Worst)"
if st.session_state.get("hl_show_min_pa", False):
    try:
        min_pa_display = int(st.session_state.get("hl_min_pa", 0))
    except Exception:
        min_pa_display = 0
    title += f" (min {min_pa_display} PA)"

grid_html = f"""
<div class="leaderboard-card">
    <div class="leaderboard-title">{title}</div>
    <div class="players-grid">
        {''.join(cards)}
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
    padding: 3rem 4rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin: 0 auto;
    width: 100%;
    max-width: 900px;
    box-sizing: border-box;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.1rem;
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
    font-size: 1.15rem;
    margin-top: 0.25rem;
}}
html, body {{
    margin: 0px;
    padding: 0px;
    background: transparent;
    width: 100%;
}}
.footer {{
    display: flex;
    justify-content: space-evenly;
    gap: 25rem;
    margin-top: 1rem;
}}
.footer p {{
    margin: 0;
    font-size: 0.9rem;
    color: #666;
    font-family: "Source Sans Pro";
}}
</style>
</head>
<body>
{grid_html}
</body>
</html>
"""

rows = max(1, (len(cards) + 4) // 5)
height = 8000

with col2:
    components.html(full_html, height=height)

# MLBAM overrides section - always exactly 10 players (2 rows of 5)
if not df.empty:
    st.markdown("---")
    st.write("Manual MLBAM overrides (enter MLBAM id to fix headshot)")
    
    # First row of 5
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
    
    # Second row of 5
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
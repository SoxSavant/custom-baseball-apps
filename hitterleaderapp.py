import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import html
import requests
from bs4 import BeautifulSoup
from pybaseball.statcast_fielding import statcast_outs_above_average
import io
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.set_page_config(layout="wide")
# ----------------------------
#  Helpers
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

    # Athletics → year-aware
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


def compute_team_display(teams: list[str]) -> str:
    if not teams:
        return "N/A"
    if len(teams) == 1:
        return teams[0]
    return f"{len(teams)} Teams"


# ----------------------------
#  External Data Loaders
# ----------------------------

@st.cache_data(ttl=900, show_spinner=False)
def batting_stats(start_year: int, end_year: int, qual=0, split_seasons=False):
    try:
        return pybaseball.batting_stats(start_year, end_year, qual=qual, split_seasons=split_seasons)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_year(year: int):
    try:
        return pybaseball.batting_stats(year, year, qual=0, split_seasons=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_bwar(start_year: int, end_year: int) -> pd.DataFrame:
    try:
        df = pybaseball.bwar_bat()
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    name_col = None
    for c in ("Name", "name", "player_name"):
        if c in df.columns:
            name_col = c
            break

    df["Name"] = df.get(name_col, "").astype(str).str.strip()
    df["NameKey"] = df["Name"].apply(normalize_statcast_name)

    if "year_ID" in df.columns:
        df["year_ID"] = pd.to_numeric(df["year_ID"], errors="coerce")
    elif "Year" in df.columns:
        df["year_ID"] = pd.to_numeric(df["Year"], errors="coerce")
    else:
        df["year_ID"] = np.nan

    war_col = None
    for c in ("WAR", "war", "War"):
        if c in df.columns:
            war_col = c
            break
    df["WAR"] = pd.to_numeric(df[war_col], errors="coerce") if war_col else np.nan

    mask = df["year_ID"].between(start_year, end_year)
    df = df[mask]

    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_bwar_span(start_year: int, end_year: int, target_names=None):
    try:
        df = load_bwar(start_year, end_year)
        if df is None or df.empty:
            return pd.DataFrame()

        df["NameKey"] = df["Name"].astype(str).apply(normalize_statcast_name)

        if target_names:
            keys = [normalize_statcast_name(n) for n in target_names]
            df = df[df["NameKey"].isin(keys)]

        # Rename to bWAR for consistency
        df["bWAR"] = df["WAR"]
        return df[["NameKey", "bWAR"]].copy()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=900)
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


@st.cache_data(show_spinner=False, ttl=900)
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


@st.cache_data(ttl=900, show_spinner=False)
def load_statcast_fielding_span(start_year: int, end_year: int, target_names=None):
    # Collect FRV (Savant) and OAA (statcast) across the span and merge by NameKey
    frames = []
    for year in range(start_year, end_year + 1):
        frv = load_savant_frv_year(year)
        if frv is not None and not frv.empty:
            frv["Season"] = year
            frames.append(frv)
        oaa = load_savant_oaa_year(year)
        if oaa is not None and not oaa.empty:
            oaa["Season"] = year
            frames.append(oaa)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["NameKey"] = combined["NameKey"].astype(str)

    if target_names:
        keys = [normalize_statcast_name(n) for n in target_names]
        combined = combined[combined["NameKey"].isin(keys)]

    # aggregate by NameKey
    out = combined.groupby("NameKey").agg({
        "FRV": "sum",
        "ARM": "sum",
        "RANGE": "sum",
        "OAA": "sum",
    }).reset_index()
    return out


@st.cache_data(ttl=900, show_spinner=False)
def get_player_teams_fangraphs(fg_id: int, start_year: int, end_year: int):
    """
    Scrape Fangraphs batting tables and extract MLB team codes.
    Handles multiple teams in a year, ATH/OAK transition, and split-season rows.
    """
    url = f"https://www.fangraphs.com/players/x/{fg_id}/stats?season=all"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return []

    found = []

    # Scan all tables (MLB + Minors + College)
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) < 2:
                continue

            # Year check
            if not (cols[0].isdigit() and len(cols[0]) == 4):
                continue

            year = int(cols[0])
            team_raw = cols[1].strip()
            # Ensure this is a Major League row
            league = cols[2] if len(cols) > 2 else ""
            if league != "MLB":
                continue

            if team_raw not in VALID_TEAMS:
                continue

            if start_year <= year <= end_year:
                team_norm = normalize_team_code(team_raw, year)
                if team_norm:
                    found.append(team_norm)

    # Unique + alphabetical
    unique = sorted(set(found))

    # Collapse OAK + ATH -> OAK/ATH if both
    collapsed = collapse_athletics(unique)

    return collapsed


# Lightweight headshot helpers (same bases used in comparison app)
HEADSHOT_BASES = [
    # Standard silo path (real photos when they exist)
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current",
    # Generic fallback path with slash
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current",
    # Alternate path (kept last)
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}headshot/67/current",
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


@st.cache_data(show_spinner=False, ttl=3600)
def lookup_mlbam_id(full_name: str, return_bbref: bool = False):
    """Try to resolve a player's mlbam id using pybaseball's playerid_lookup.

    Returns (mlbam, bbref) or (None, None) when not found.
    """
    try:
        if not full_name or not isinstance(full_name, str):
            return (None, None) if return_bbref else None
        df = pybaseball.playerid_lookup(full_name)
        if df is None or df.empty:
            return (None, None) if return_bbref else None
        # prefer key_mlbam if present
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


@st.cache_data(show_spinner=False, ttl=86400)
def build_mlb_headshot(mlbam: int | str | None) -> str | None:
    """Build a candidate MLB static headshot URL given an mlbam id.

    This mirrors the comparison app: we don't verify the URL here to keep the
    hot path fast — the first working base is returned formatted.
    """
    if mlbam is None:
        return None
    try:
        for base in HEADSHOT_BASES:
            try:
                return base.format(mlbam=int(mlbam))
            except Exception:
                # fallback to string formatting
                try:
                    return base.format(mlbam=str(mlbam).strip())
                except Exception:
                    continue
    except Exception:
        return None
    return None


def get_headshot_url_from_row(row: pd.Series) -> str:
    """Resolve the best headshot URL for a single-row (aggregated or yearly).

    Strategy:
    1. Look for mlbam-like columns and build an MLB headshot URL.
    2. If missing, look for FanGraphs id and reverse-lookup to mlbam.
    3. As a last resort, try to look up mlbam by the player's name.
    4. Return a data-URI placeholder when nothing resolved.
    """
    # 1) direct mlbam-like columns
    for col in ("mlbam", "MLBID", "key_mlbam", "mlbam_id", "MLBAMID", "mlbam_override"):
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                url = build_mlb_headshot(val)
                if url:
                    return url

    # 2) FanGraphs id reverse lookup
    for fg_col in ("playerid", "IDfg", "fg_id", "FGID", "IDfg"):
        if fg_col in row.index:
            fg = row.get(fg_col)
            if pd.notna(fg) and str(fg).strip():
                try:
                    rev = pybaseball.playerid_reverse_lookup([int(fg)], key_type="fangraphs")
                    if rev is not None and not rev.empty:
                        mlbam = rev.iloc[0].get("key_mlbam")
                        if pd.notna(mlbam):
                            url = build_mlb_headshot(mlbam)
                            if url:
                                return url
                except Exception:
                    pass

    # 3) Try lookup by name
    name = None
    if "Name" in row.index:
        name = str(row.get("Name", "")).strip()
    elif "player_name" in row.index:
        name = str(row.get("player_name", "")).strip()

    if name:
        try:
            mlbam = lookup_mlbam_id(name)
            if mlbam:
                url = build_mlb_headshot(mlbam)
                if url:
                    return url
        except Exception:
            pass

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

    # Team aggregation
    teams = grp.get("Team", pd.Series([], dtype=str)).dropna().astype(str).tolist()
    teams = [t.strip().upper() for t in teams if t.strip()]
    teams = [normalize_team_code(t, int(grp["Season"].iloc[0]) if "Season" in grp.columns else 2025) for t in teams]
    teams = collapse_athletics(sorted(set([t for t in teams if t])))

    result["Teams"] = teams
    result["TeamDisplay"] = compute_team_display(teams)

    # Pick the most recent headshot / id info from the grouped rows so spans use
    # the latest available MLBAM or FanGraphs id for the player's picture.
    try:
        if "Season" in grp.columns:
            grp_sorted = grp.sort_values(by="Season", ascending=False)
        else:
            grp_sorted = grp.iloc[::-1]

        # MLBAM-like candidates
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
            # Use IDfg as a canonical FG id column on aggregated rows
            result["IDfg"] = found_fg
    except Exception:
        # non-critical; ignore
        pass

    # Copy other columns — skip id columns so our chosen mlbam/IDfg are not
    # overwritten by numeric sums across seasons.
    skip_cols = {
        "Name",
        "Team",
        "Season",
        "Teams",
        # mlbam / MLB id variants
        "mlbam",
        "MLBID",
        "key_mlbam",
        "mlbam_id",
        "MLBAMID",
        # FanGraphs id variants
        "playerid",
        "IDfg",
        "fg_id",
        "FGID",
    }

    for col in grp.columns:
        if col in skip_cols:
            continue
        try:
            result[col] = grp[col].sum()
        except Exception:
            result[col] = grp[col].iloc[0]

    # Derive rate stats from aggregated counting stats where appropriate.
    def to_num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    # Ensure numeric conversions for common counting stats
    h = to_num(result.get("H"))
    ab = to_num(result.get("AB"))
    bb = to_num(result.get("BB"))
    hbp = to_num(result.get("HBP"))
    sf = to_num(result.get("SF"))
    doubles = to_num(result.get("2B"))
    triples = to_num(result.get("3B"))
    hr = to_num(result.get("HR"))

    # Derive singles and TB when possible
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

    # AVG
    if pd.notna(ab) and ab > 0 and pd.notna(h):
        result["AVG"] = h / ab
    else:
        result["AVG"] = np.nan

    # SLG
    if pd.notna(ab) and ab > 0 and pd.notna(tb):
        result["SLG"] = tb / ab
    else:
        result["SLG"] = np.nan

    # OBP
    bb_val = 0 if pd.isna(bb) else bb
    hbp_val = 0 if pd.isna(hbp) else hbp
    sf_val = 0 if pd.isna(sf) else sf
    obp_den = (ab if pd.notna(ab) else 0) + bb_val + hbp_val + sf_val
    if obp_den > 0 and pd.notna(h):
        result["OBP"] = (h + bb_val + hbp_val) / obp_den
    else:
        result["OBP"] = np.nan

    # ISO when possible
    if pd.notna(result.get("SLG")) and pd.notna(result.get("AVG")):
        try:
            result["ISO"] = float(result.get("SLG")) - float(result.get("AVG"))
        except Exception:
            result["ISO"] = np.nan
    else:
        result["ISO"] = np.nan

    # For other rate stats, compute PA-weighted averages when PA is available
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

    # Compute PA-weighted averages using the original group rows when the
    # stat is present on the per-year rows. This prevents summed values for
    # rate stats across multi-year spans (e.g., HardHit%, EV, wRC+, wOBA,
    # xwOBA, xBA, xSLG, OPS). If the per-row stat isn't available but the
    # function already derived a value (e.g., AVG/OBP/SLG from counting stats),
    # keep that derived value.
    if pa_total > 0:
        for rs in rate_stats:
            # Find a matching per-row column case-insensitively (e.g., WRC+ vs wRC+)
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
                        # assign to the actual column name so any previously
                        # summed value is overwritten
                        result[matching_col] = np.nan
                    else:
                        # assign to the actual column name so any previously
                        # summed value is overwritten
                        result[matching_col] = numer / pa_total
                except Exception:
                    result[matching_col] = np.nan
            else:
                # If the stat wasn't present per-row, prefer any already-derived
                # value (like AVG from counts). Otherwise set NaN.
                if rs in result and pd.notna(result.get(rs)):
                    continue
                result[rs] = np.nan

    return result


# ----------------------------
#  Enrichment
# ----------------------------

SUM_STATS = {"HR", "R", "RBI", "SB", "BB", "H", "PA", "AB"}


def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""

    upper_stat = stat.upper()
    if upper_stat == "FRV":
        return f"{int(round(float(val)))}"

    if upper_stat == "ARM":
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
    """
    Normalize or derive stat values before formatting/comparison.
    Whiff% is not provided directly, so derive it from Contact% (100 - Contact%).
    """
    if stat == "Contact%":
        if pd.isna(raw_val):
            return np.nan
        try:
            contact = float(raw_val)
        except Exception:
            return np.nan
        # Contact% may come in as a fraction (0.78) or percentage (78.0).
        if contact <= 1:
            contact *= 100
        return 100 - contact
    return raw_val

def enrich_leaderboard_players(start_year: int, end_year: int, df: pd.DataFrame, combined: pd.DataFrame | None = None, resolve_fg_teams: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    if "Name" not in out.columns:
        return out

    out["NameKey"] = out["Name"].astype(str).apply(normalize_statcast_name)

    if combined is not None and not combined.empty and "NameKey" not in combined.columns:
        combined = combined.copy()
        combined["NameKey"] = combined["Name"].astype(str).apply(normalize_statcast_name)

    # bWAR merge
    try:
        names = tuple(out["Name"].dropna().unique())
        bwar = load_bwar_span(start_year, end_year, target_names=names)
        if not bwar.empty:
            out = pd.merge(out, bwar, on="NameKey", how="left")
        else:
            out["bWAR"] = np.nan
    except Exception:
        out["bWAR"] = np.nan

    # Fielding merge
    try:
        names = tuple(out["Name"].dropna().unique())
        field = load_statcast_fielding_span(start_year, end_year, target_names=names)
        if not field.empty:
            out = pd.merge(out, field, on="NameKey", how="left")
        else:
            for m in ["FRV", "OAA", "ARM", "RANGE"]:
                out[m] = np.nan
    except Exception:
        for m in ["FRV", "OAA", "ARM", "RANGE"]:
            out[m] = np.nan

    out["Team"] = out.get("Team", "")
    out["TeamDisplay"] = out.get("TeamDisplay", "")

    for idx, prow in out.iterrows():
        name_key = prow.get("NameKey")
        if not name_key:
            out.at[idx, "TeamDisplay"] = "N/A"
            continue

        fg_id = prow.get("IDfg", None)
        if resolve_fg_teams and pd.notna(fg_id):
            try:
                fg_id = int(fg_id)
                vals = get_player_teams_fangraphs(fg_id, start_year, end_year)
                if vals:
                    collapsed = collapse_athletics(sorted(set([t for t in vals if t])))
                    out.at[idx, "TeamDisplay"] = compute_team_display(collapsed)
                    out.at[idx, "Team"] = ",".join(collapsed)
                    continue
            except Exception:
                pass

        if combined is None or combined.empty:
            out.at[idx, "TeamDisplay"] = "N/A"
            continue

        sub = combined[combined["NameKey"] == name_key]
        if sub.empty:
            out.at[idx, "TeamDisplay"] = "N/A"
            continue

        teams = sub.get("Team", pd.Series([], dtype=str)).dropna().astype(str).tolist()
        teams = [t.strip().upper() for t in teams if t.strip()]
        teams = [normalize_team_code(t, int(sub["Season"].iloc[0]) if "Season" in sub.columns else 2025) for t in teams]
        teams = collapse_athletics(sorted(set([t for t in teams if t])))

        out.at[idx, "TeamDisplay"] = compute_team_display(teams)
        out.at[idx, "Team"] = ",".join(teams)

    return out


# ----------------------------
#  Load batting
# ----------------------------

@st.cache_data(ttl=900, show_spinner=False)
def load_batting(start_year: int, end_year: int, resolve_fg_teams: bool = False) -> pd.DataFrame:
    start = min(start_year, end_year)
    end = max(start_year, end_year)

    if start == end:
        df = batting_stats(start, end, qual=0, split_seasons=False)
        if df.empty:
            df = load_year(start)
        if df.empty:
            return pd.DataFrame()
        return enrich_leaderboard_players(start, end, df, combined=df, resolve_fg_teams=resolve_fg_teams)

    frames = []
    failed_years = []
    for year in range(start, end + 1):
        yearly = batting_stats(year, year, qual=0, split_seasons=False)
        if yearly.empty:
            yearly = load_year(year)
        if yearly.empty:
            failed_years.append(year)
            continue
        frames.append(yearly)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["NameKey"] = combined["Name"].astype(str).apply(normalize_statcast_name)

    grouped_rows = []
    for name, grp in combined.groupby("Name"):
        grouped_rows.append(aggregate_player_group(grp, name))

    aggregated = pd.DataFrame(grouped_rows)
    aggregated = enrich_leaderboard_players(start, end, aggregated, combined=combined, resolve_fg_teams=resolve_fg_teams)

    if failed_years:
        st.info(f"Loaded partial data; skipped years: {', '.join(map(str, failed_years))}")

    return aggregated


# ----------------------------
#  UI
# ----------------------------

STAT_ALLOWLIST = [
    "Off", "Def", "BsR", "WAR", "bWAR", "Barrel%", "HardHit%", "EV",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "H", "1B", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Z-Swing%", "Swing%", "Contact%", "WPA", "Clutch",
    "Pull%", "Cent%", "Oppo%", "GB%", "FB%", "LD%", "LA", "FRV", "OAA", "ARM", "DRS", "TZ",
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
    st.title("Custom Hitter Leaderbord")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# default to the current year as a single-year view
current_year = date.today().year
if "hl_start_year" not in st.session_state:
    # default to 2025 per app convention
    st.session_state["hl_start_year"] = 2025
if "hl_end_year" not in st.session_state:
    st.session_state["hl_end_year"] = 2025
if "hl_stat" not in st.session_state:
    st.session_state["hl_stat"] = "WAR"

def compute_default_pa(start: int, end: int) -> int:
    # keep helper but default behavior for the app is a flat 502 PA minimum
    return 502

def on_year_change():
    s = st.session_state
    start = int(s.get("hl_start_year", 2025))
    end = int(s.get("hl_end_year", 2025))
    # keep the single-year / span sync: when not in span mode keep end == start
    if not s.get("hl_span", False):
        s["hl_end_year"] = s["hl_start_year"]

def on_stat_change():
    s = st.session_state
    start = int(s.get("hl_start_year", 2025))
    end = int(s.get("hl_end_year", 2025))
    # do not change the user's chosen Min PA when switching stats
    # keep span sync as well
    if not s.get("hl_span", False):
        s["hl_end_year"] = s["hl_start_year"]

if "hl_min_pa" not in st.session_state:
    # default minimum plate appearances is 502 (single-season convention)
    st.session_state["hl_min_pa"] = 502

st.markdown(
    """
    <style>
        /* shrink dropdowns and number inputs in the controls column */
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
    on_change=on_stat_change,
    format_func=lambda x: label_map.get(x, x),
)

if "hl_span" not in st.session_state:
    # single-year view by default; checkbox enables multi-year span
    st.session_state["hl_span"] = False

col1, col2 = st.columns([.5, 2])

with col1:
    st.checkbox("Multi-year span", key="hl_span", on_change=on_year_change)
    start_label = "Start Year" if st.session_state.get("hl_span", False) else "Year"
    start_year = st.number_input(
        start_label,
        min_value=1900,
        max_value=current_year,
        value=st.session_state["hl_start_year"],
        key="hl_start_year",
        on_change=on_year_change
    )
    # only show the end-year when the user has requested a multi-year span
    if st.session_state.get("hl_span", False):
        end_year = st.number_input(
            "End Year",
            min_value=1900,
            max_value=current_year,
            value=st.session_state["hl_end_year"],
            key="hl_end_year",
            on_change=on_year_change
        )
    else:
        # ensure end == start for single-year default and keep a local var for loads
        end_year = st.session_state["hl_start_year"]
    # Minimum plate appearances (default 502) shown under the year selectors
    min_pa = st.number_input("Min PA", min_value=0, max_value=20000, value=st.session_state["hl_min_pa"], key="hl_min_pa")

    st.checkbox("Show worst instead", key="hl_sort_worst")
    st.checkbox("Show min PA", key="hl_show_min_pa")

df = load_batting(int(start_year), int(end_year))
if df.empty:
    st.error("No data loaded.")
    st.stop()

# Apply Min PA filter (ensure numeric PA)
try:
    min_pa_val = int(st.session_state.get("hl_min_pa", 0))
except Exception:
    min_pa_val = 0
if "PA" in df.columns:
    df["PA"] = pd.to_numeric(df["PA"], errors="coerce").fillna(0)
    df = df[df["PA"] >= min_pa_val]

# Sort and take top N
df = df.sort_values(by=stat, ascending=st.session_state.get("hl_sort_worst", False))
df = df.head(10)

# Resolve FanGraphs team names only for the top-10 rows to keep loading fast.
try:
    with st.spinner("Resolving teams for top players..."):
        for idx in df.index:
            try:
                fg_id = df.at[idx, "IDfg"] if "IDfg" in df.columns else None
                if pd.notna(fg_id):
                    fg_int = int(fg_id)
                    vals = get_player_teams_fangraphs(fg_int, int(start_year), int(end_year))
                    if vals:
                        collapsed = collapse_athletics(sorted(set([t for t in vals if t])))
                        df.at[idx, "TeamDisplay"] = compute_team_display(collapsed)
                        df.at[idx, "Team"] = ",".join(collapsed)
                # Ensure mlbam override exists for headshot resolution (fix accents/rookies)
                name_val = df.at[idx, "Name"] if "Name" in df.columns else None
                if name_val and ("mlbam_override" not in df.columns or pd.isna(df.at[idx, "mlbam_override"])):
                    try:
                        mlbam = lookup_mlbam_id(str(name_val))
                        if mlbam:
                            df.at[idx, "mlbam_override"] = mlbam
                    except Exception:
                        # non-fatal; continue
                        pass
            except Exception:
                # non-fatal; leave existing Team/TeamDisplay as-is
                continue
except Exception:
    # If spinner or network calls fail, proceed without blocking the UI
    pass

cards = []
for _, row in df.iterrows():
    name = row.get("Name", "")
    team = row.get("TeamDisplay", "")
    raw_val = row.get(stat, np.nan)
    transformed = transform_stat_value(stat, raw_val)
    display_val = format_stat(stat, transformed)
    # Honor any manual MLBAM override entered by the user (widget keys hl_mlbam_override_0..9).
    # We read from st.session_state so the input can be rendered later in the page but
    # still affect headshot resolution on rerun.
    src_row = row
    try:
        # find the index of this row within the current top-10 (0..9)
        pos = list(df.index).index(row.name)
        key = f"hl_mlbam_override_{pos}"
        override_val = st.session_state.get(key, "")
        if override_val is not None and str(override_val).strip():
            try:
                ov = int(str(override_val).strip())
                src_row = row.copy()
                src_row["mlbam_override"] = ov
            except Exception:
                # ignore invalid overrides
                pass
    except Exception:
        # fallback: no session override available
        pass

    src = get_headshot_url_from_row(src_row)
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

# ...existing code...

span_label = f"{int(start_year)}" if start_year == end_year else f"{int(start_year)}–{int(end_year)}"
title_label = label_map.get(stat, stat)
title = f"{span_label} {title_label} Leaders"
if st.session_state.get("hl_sort_worst", False):
    title += " (Worst)"
# Optionally show the minimum PA used for filtering in the title
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
    padding: 1.3rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    /* left-align inside the column instead of centering page-wide */
    margin: 0;
    width: 100%;
    max-width: none;
    box-sizing: border-box;
}}
.leaderboard-title {{
    font-weight: 900;
    font-size: 2.1rem;
    margin-bottom: 1rem;
    text-align: center;
}}
.players-grid {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    /* align items to the left so the grid doesn't get centered */
    justify-content: start;
    row-gap: 1rem;
}}
.player-card {{
    text-align: center;
}}
.player-card img {{
    width: 140px;
    height: 140px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    background: #f6f6f6;
}}
.player-name {{
    font-weight: 800;
    margin-top: 0.25rem;
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
height = min(2200, 260 * rows + 180)

with col2:
    components.html(full_html, height=height)

# Manual MLBAM id overrides for displayed players (rendered in a full-width row
# below the controls and cards so the leaderboard grid isn't squeezed)
if not df.empty:
    st.markdown("---")
    st.write("Manual MLBAM overrides (enter MLBAM id to fix headshot)")
    cols_row = [st.columns(5), st.columns(5)]
    for i, idx in enumerate(df.index):
        row_cols = cols_row[i // 5]
        with row_cols[i % 5]:
            key = f"hl_mlbam_override_{i}"
            default_val = ""
            if "mlbam_override" in df.columns and pd.notna(df.at[idx, "mlbam_override"]):
                try:
                    default_val = str(int(df.at[idx, "mlbam_override"]))
                except Exception:
                    default_val = str(df.at[idx, "mlbam_override"]) if pd.notna(df.at[idx, "mlbam_override"]) else ""
            user_val = st.text_input(f"Player {i+1} MLBAM", value=default_val, key=key)
            try:
                if user_val and str(user_val).strip():
                    df.at[idx, "mlbam_override"] = int(str(user_val).strip())
                else:
                    df.at[idx, "mlbam_override"] = np.nan
            except Exception:
                df.at[idx, "mlbam_override"] = np.nan

import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import html
import requests
import re
from bs4 import BeautifulSoup
import io
from pathlib import Path
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.set_page_config(layout="wide")

# ----------------------------
#  MEMORY-OPTIMIZED DATA LOADING
# ----------------------------

@st.cache_data(ttl=3600, max_entries=3)
def load_filtered_data(year, year2, min_ip=0):
    """
    Load data and filter by IP threshold.
    For single year: use qual parameter (much faster!)
    For multi-year: filter AFTER aggregation (so min_ip represents total IP across all years).
    """
    df = pitching_stats(year, year2, qual=min_ip, split_seasons=False)
        
       
    if not df.empty and "Team" in df.columns:
        def make_team_display(team_val):
            if pd.isna(team_val):
                return "N/A"
            team_str = str(team_val).strip()
                # FanGraphs uses these patterns for multi-team players
            if team_str in {"---", "- - -", "--", "TOT", ""}:
                return "2 Teams"
                # Otherwise normalize the team code
            normalized = normalize_team_code(team_str, year)
            return normalized if normalized else "N/A"
            
        df["TeamDisplay"] = df["Team"].apply(make_team_display)
        
        return df


def optimize_dtypes(df):
    """Convert data types to use less memory"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Convert float64 to float32 where appropriate
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        # Keep high precision for rate stats
        if col not in ['ERA', 'FIP', 'xFIP', 'WHIP', 'K/9', 'BB/9']:
            df[col] = df[col].astype('float32')
    
    # Convert int64 to int32 where appropriate
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() < 2147483647:  # int32 max
            df[col] = df[col].astype('int32')
    
    return df


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

@st.cache_data(ttl=600, max_entries=2)
def pitching_stats(start_year: int, end_year: int, qual=0, split_seasons=False):
    try:
        return pybaseball.pitching_stats(start_year, end_year, qual=qual, split_seasons=split_seasons)
    except Exception:
        return pd.DataFrame()


# Lightweight headshot helpers
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
    """Best-effort MLBAM lookup using pybaseball's playerid_lookup. Optionally returns bbref id."""
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
    """Try MLB headshot URLs in order; return the first that responds (200)."""
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
    """Best-effort guesses for bbref slug when lookup fails."""
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
    """Comprehensive headshot resolution with multiple fallback strategies."""
    name = str(row.get("Name", "")).strip()
    
    # 1) Try direct MLBAM columns
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
    
    # 2) Try FanGraphs ID reverse lookup
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
    
    # 3) Try direct BBRef columns
    bref_cols = ["key_bbref", "bbref_id", "BBREFID", "bref_id", "BREFID"]
    for col in bref_cols:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                bref_url = resolve_bref_headshot(str(val))
                if bref_url:
                    return bref_url
    
    # 4) Try name-based lookup
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
    
    # 5) Try heuristic BBRef slugs
    if name:
        for slug in heuristic_bbref_slug(name):
            bref_url = resolve_bref_headshot(slug)
            if bref_url:
                return bref_url
    
    # 6) Fallback to placeholder
    return HEADSHOT_PLACEHOLDER


# ----------------------------
#  Aggregation
# ----------------------------

def ip_to_outs(value) -> float:
    """Convert MLB innings notation (e.g., 5.1/5.2) to outs."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan

    if isinstance(value, str):
        match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", value)
        if not match:
            return np.nan
        try:
            v = float(match.group(0))
        except Exception:
            return np.nan
    else:
        try:
            v = float(value)
        except Exception:
            return np.nan

    innings = int(np.floor(v))
    fractional = v - innings
    if abs(fractional - 0.1) < 0.05:
        outs_extra = 1
    elif abs(fractional - 0.2) < 0.05:
        outs_extra = 2
    else:
        outs_extra = int(round(fractional * 3))
        outs_extra = min(max(outs_extra, 0), 2)
    return innings * 3 + outs_extra


def outs_to_ip(outs: float) -> float:
    """Convert outs back to MLB innings notation."""
    if pd.isna(outs):
        return np.nan
    total_outs = float(outs)
    innings = int(total_outs // 3)
    remainder = int(round(total_outs % 3))
    return innings + remainder / 10


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

    # Pitching-specific aggregation
    SUM_STATS = {"G", "GS", "W", "L", "SV", "IP", "SO", "BB", "HR", "ER", "WAR"}
    RATE_STATS = {"ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "K%", "BB%", "K-BB%", 
                  "Barrel%", "HardHit%", "EV", "O-Swing%", "Contact%", "GB%", "FB%"}

    for col in grp.columns:
        if col in skip_cols:
            continue

        series = pd.to_numeric(grp[col], errors="coerce")
        
        # Handle IP specially
        if col == "IP":
            outs_series = series.apply(ip_to_outs)
            valid_outs = outs_series.dropna()
            if valid_outs.empty:
                continue
            ip_outs_total = valid_outs.sum()
            result[col] = outs_to_ip(ip_outs_total)
            continue

        if series.isna().all():
            continue

        # Sum stats vs rate stats
        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in RATE_STATS:
            # Weight by IP or TBF
            if "IP" in grp.columns:
                weight = pd.to_numeric(grp["IP"], errors="coerce").fillna(0)
            elif "TBF" in grp.columns:
                weight = pd.to_numeric(grp["TBF"], errors="coerce").fillna(0)
            else:
                weight = pd.Series(np.ones(len(grp)), index=grp.index)
            
            weight_total = weight.sum()
            if weight_total > 0:
                result[col] = (series * weight).sum(skipna=True) / weight_total
            else:
                result[col] = series.mean(skipna=True)
        else:
            # Default: sum
            try:
                result[col] = series.sum(skipna=True)
            except Exception:
                result[col] = grp[col].iloc[0]

    return result


# ----------------------------
#  Formatting
# ----------------------------

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""

    upper_stat = stat.upper()

    if upper_stat == "AGE":
        if isinstance(val, str):
            return val
        v = float(val)
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"WAR", "FWAR", "EV"}:
        v = float(val)
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}.0"
        return f"{v:.1f}"

    if upper_stat in {"ERA", "FIP", "XFIP", "K/9", "BB/9", "HR/9", "XERA",}:
        return f"{float(val):.2f}"

    if upper_stat == "WHIP":
        return f"{float(val):.3f}"

    if upper_stat == "IP":
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"ERA-", "FIP-"}:
        return f"{int(round(float(val)))}"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat or "BB%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"

    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def transform_stat_value(stat: str, raw_val):
    """Transform stat values if needed (e.g., Contact% to Whiff%)"""
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
    "WAR", "ERA", "xERA", "FIP", "xFIP", "IP", "G", "GS", "W", "L", "SV", "SO", "BB", "K/9", "BB/9",
    "HR/9", "K%", "BB%", "K-BB%", "WHIP", "ERA-", "FIP-", "Barrel%", "HardHit%", "EV",
    "O-Swing%", "Contact%", "GB%", "FB%", "CG", "ShO"
]

label_map = {
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
    "EV": "Avg Exit Velo",
    "O-Swing%": "Chase%",
    "Contact%": "Whiff%",
}

lower_better = {
    "HardHit%", "Barrel%", "EV", "ERA", "xERA", "FIP", "xFIP", "BB", "HBP", "HR",
    "BB/9", "HR/9", "BABIP", "HR/FB", "BB%", "AVG", "WHIP", "ERA-", "FIP-",
    "FB%", "SIERA", "Z-Swing%", "Contact%", "Pull%", "LD%", "L"
}

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Custom Pitcher Leaderboard")
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
if "pl_start_year" not in st.session_state:
    st.session_state["pl_start_year"] = 2025
if "pl_end_year" not in st.session_state:
    st.session_state["pl_end_year"] = 2025
if "pl_stat" not in st.session_state:
    st.session_state["pl_stat"] = "WAR"
if "pl_min_ip" not in st.session_state:
    st.session_state["pl_min_ip"] = 162

def on_year_change():
    s = st.session_state
    if not s.get("pl_span", False):
        s["pl_end_year"] = s["pl_start_year"]

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
    key="pl_stat",
    format_func=lambda x: label_map.get(x, x),
)

if "pl_span" not in st.session_state:
    st.session_state["pl_span"] = False

col1, col2 = st.columns([.5, 2])

with col1:
    
    start_label = "Start Year" if st.session_state.get("pl_span", False) else "Year"
    year = st.number_input(
        start_label,
        min_value=1900,
        max_value=current_year,
        key="pl_start_year",
        on_change=on_year_change
    )
   
    min_ip = st.number_input(
        "Min IP",
        min_value=0,
        max_value=5000,
        key="pl_min_ip"
    )
    st.checkbox("Show worst", key="pl_sort_worst")
    st.checkbox("Show min IP", key="pl_show_min_ip")

# Load filtered data
min_ip_val = int(st.session_state.get("pl_min_ip", 0))
df = load_filtered_data(year, year, min_ip_val)

if st.checkbox("Show available columns"):
            st.write("All columns:", sorted(df.columns.tolist()))
            st.write("Percentage stats:", [c for c in df.columns if '%' in c])

# Sort and limit to top 10
if not df.empty and stat in df.columns:
    # For lower_better stats, we want ascending=True when NOT showing worst
    # For higher_better stats, we want ascending=False when NOT showing worst
    is_lower_better = stat in lower_better
    show_worst = st.session_state.get("pl_sort_worst", False)
    
    if is_lower_better:
        # For ERA, FIP, etc.: ascending=True shows best (lowest), ascending=False shows worst (highest)
        df = df.sort_values(by=stat, ascending=not show_worst)
    else:
        # For WAR, SO, etc.: ascending=False shows best (highest), ascending=True shows worst (lowest)
        df = df.sort_values(by=stat, ascending=show_worst)
    
    df = df.head(10)
else:
    if not df.empty:
        st.error(f"Column '{stat}' not found. Available columns: {', '.join(df.columns)}")
    df = pd.DataFrame()

# Ensure TeamDisplay exists
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
        key = f"pl_mlbam_override_{pos}"
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

span_label = f"{int(year)}"
title_label = label_map.get(stat, stat)
title = f"{span_label} {title_label} Leaders"
if st.session_state.get("pl_sort_worst", False):
    title += " (Worst)"
if st.session_state.get("pl_show_min_ip", False):
    try:
        min_ip_display = int(st.session_state.get("pl_min_ip", 0))
    except Exception:
        min_ip_display = 0
    title += f" (min {min_ip_display} IP)"

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
    margin: -1rem auto 0 auto;
    margin-top: 0;
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

with col2:
    components.html(full_html, height=800)

# MLBAM overrides section
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
            key = f"pl_mlbam_override_{player_idx}"
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
            key = f"pl_mlbam_override_{player_idx}"
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
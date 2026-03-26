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
import io
from pathlib import Path
from datetime import date
import streamlit.components.v1 as components
import pybaseball

st.error("⚠️ Data source temporarily down. Working on a fix.")

st.set_page_config(page_title="Custom Pitching Leaderboard", layout="wide", page_icon="⚾",)

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

STATCAST_START_YEAR = 2015
STATCAST_RATE_STATS = {"Barrel%", "HardHit%", "EV"}

# ----------------------------
#  MEMORY-OPTIMIZED DATA LOADING
# ----------------------------

@st.cache_data(ttl=3600, max_entries=3)
def load_filtered_data(start_year, end_year, min_ip=0):
    if start_year == end_year:
        df = pitching_stats(start_year, end_year, qual=min_ip, split_seasons=False)

        if not df.empty and "Team" in df.columns:
            def make_team_display(team_val):
                if pd.isna(team_val):
                    return "N/A"
                team_str = str(team_val).strip()
                if team_str in {"---", "- - -", "--", "TOT", ""}:
                    return "2 Teams"
                normalized = normalize_team_code(team_str, start_year)
                return normalized if normalized else "N/A"
            df["TeamDisplay"] = df["Team"].apply(make_team_display)

        return df
    else:
        num_years = end_year - start_year + 1

        frames = []
        for year in range(start_year, end_year + 1):
            yr_data = pitching_stats(year, year, qual=1, split_seasons=False)
            if not yr_data.empty:
                yr_data = yr_data.copy()
                yr_data["Season"] = year
                frames.append(yr_data)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        if min_ip > 0:
            single_season_threshold = min_ip / num_years * 0.5
            ip_per_row = pd.to_numeric(combined["IP"], errors="coerce").fillna(0)
            qualifying_mask = ip_per_row >= single_season_threshold
            candidate_names = set(combined.loc[qualifying_mask, "Name"].tolist())
            combined = combined[combined["Name"].isin(candidate_names)]

        combined = optimize_dtypes(combined)

        grouped_rows = []
        for player_id, grp in combined.groupby("IDfg"):
            name = grp["Name"].iloc[0] if not grp.empty else None
            row = aggregate_player_group(grp, name, start_year=start_year)
            if row is not None and len(row):
                grouped_rows.append(row)

        result = pd.DataFrame(grouped_rows)
        result = optimize_dtypes(result)

        if not result.empty and min_ip > 0 and "IP" in result.columns:
            result = result[pd.to_numeric(result["IP"], errors="coerce").fillna(0) >= min_ip]

        return result


# ----------------------------
#  SPLIT SEASON DATA LOADING
# ----------------------------

@st.cache_data(ttl=3600, max_entries=3)
def load_split_season_data(start_year, end_year, min_ip=0):
    """One row per pitcher-season — best individual season in the span."""
    frames = []
    for year in range(start_year, end_year + 1):
        yr_data = pitching_stats(year, year, qual=min_ip, split_seasons=False)
        if yr_data is None or yr_data.empty:
            continue
        yr_data = yr_data.copy()

        if "Team" in yr_data.columns:
            def make_team_display(team_val):
                if pd.isna(team_val):
                    return "N/A"
                team_str = str(team_val).strip()
                if team_str in {"---", "- - -", "--", "TOT", ""}:
                    return "2 Teams"
                normalized = normalize_team_code(team_str, year)
                return normalized if normalized else "N/A"
            yr_data["TeamDisplay"] = yr_data["Team"].apply(make_team_display)

        yr_data["Season"] = year
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
        if col not in {"ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9"}:
            df[col] = df[col].astype("float32")
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        if df[col].max() < 2147483647:
            df[col] = df[col].astype("int32")
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


def compute_team_display(teams: list[str]) -> str:
    if not teams:
        return "N/A"
    if len(teams) == 1:
        return teams[0]
    return f"{len(teams)} Teams"


# ----------------------------
#  External Data Loaders
# ----------------------------

@st.cache_data(ttl=3600, max_entries=2)
def pitching_stats(start_year: int, end_year: int, qual=0, split_seasons=False):
    try:
        return pybaseball.pitching_stats(start_year, end_year, qual=qual, split_seasons=split_seasons)
    except Exception:
        return pd.DataFrame()


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


@st.cache_data(show_spinner=False, ttl=3600)
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
    return [f"{base_slug}{i:02d}" for i in range(1, 16)]


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

def ip_to_outs(value) -> float:
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
    if pd.isna(outs):
        return np.nan
    total_outs = float(outs)
    innings = int(total_outs // 3)
    remainder = int(round(total_outs % 3))
    return innings + remainder / 10


SUM_STATS = {"G", "GS", "W", "L", "SV", "HLD", "BS", "SO", "BB", "HR", "ER", "H", "HBP",
             "IBB", "WP", "BK", "R", "WAR", "CG", "ShO", "TBF"}
RATE_STATS = {"FIP", "xFIP", "xERA", "WHIP", "K/9", "BB/9", "HR/9", "K%", "BB%",
              "K-BB%", "Barrel%", "HardHit%", "EV", "O-Swing%", "Contact%", "GB%", "FB%",
              "LD%", "HR/FB", "BABIP", "SIERA", "ERA-", "FIP-"}


def aggregate_player_group(grp: pd.DataFrame, name: str | None = None, start_year: int = 2015) -> dict:
    result: dict[str, object] = {}

    if name is None and "Name" in grp.columns:
        val = grp["Name"].dropna()
        if not val.empty:
            name = str(val.iloc[0])
    if name:
        result["Name"] = name

    teams = grp.get("Team", pd.Series([], dtype=str)).dropna().astype(str).tolist()
    teams = [t.strip().upper() for t in teams if t.strip()]
    ref_year = int(grp["Season"].iloc[0]) if "Season" in grp.columns else 2025
    teams = [normalize_team_code(t, ref_year) for t in teams]
    teams = collapse_athletics(sorted(set([t for t in teams if t])))
    result["Teams"] = teams
    result["TeamDisplay"] = compute_team_display(teams)

    try:
        grp_sorted = grp.sort_values(by="Season", ascending=False) if "Season" in grp.columns else grp.iloc[::-1]
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

    if "IP" in grp.columns:
        ip_series = pd.to_numeric(grp["IP"], errors="coerce").fillna(0)
        weight = ip_series
    elif "TBF" in grp.columns:
        weight = pd.to_numeric(grp["TBF"], errors="coerce").fillna(0)
    else:
        weight = pd.Series(np.ones(len(grp)), index=grp.index)
    weight_total = weight.sum()

    if "IP" in grp.columns:
        ip_num = pd.to_numeric(grp["IP"], errors="coerce")
        outs_series = ip_num.apply(ip_to_outs)
        valid_outs = outs_series.dropna()
        if not valid_outs.empty:
            result["IP"] = outs_to_ip(valid_outs.sum())

    for col in grp.columns:
        if col in skip_cols or col == "IP":
            continue

        series = pd.to_numeric(grp[col], errors="coerce")
        if series.isna().all():
            continue

        if col == "Age":
            age_min = series.min(skipna=True)
            age_max = series.max(skipna=True)
            if pd.isna(age_min) or pd.isna(age_max):
                continue
            if abs(age_min - age_max) < 0.01:
                result[col] = float(age_min)
            else:
                result[col] = f"{int(round(age_min))}-{int(round(age_max))}"
            continue

        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in RATE_STATS and weight_total > 0:
            if col in STATCAST_RATE_STATS:
                if start_year >= STATCAST_START_YEAR:
                    result[col] = (series * weight).sum(skipna=True) / weight_total
                else:
                    result[col] = np.nan
            else:
                result[col] = (series * weight).sum(skipna=True) / weight_total
        else:
            result[col] = series.mean(skipna=True)

    er_val = result.get("ER")
    ip_val = result.get("IP")
    if pd.notna(er_val) and pd.notna(ip_val):
        ip_outs = ip_to_outs(ip_val)
        ip_innings = ip_outs / 3 if pd.notna(ip_outs) and ip_outs > 0 else None
        if ip_innings:
            result["ERA"] = (float(er_val) / ip_innings) * 9

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

    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "K/9", "BB/9", "HR/9"}:
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

# ----------------------------
#  Page header
# ----------------------------
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

# ----------------------------
#  Session state defaults
# ----------------------------
for key, default in [
    ("pl_year",        2025),
    ("pl_start_year",  2024),
    ("pl_end_year",    2025),
    ("pl_stat",        "WAR"),
    ("pl_min_ip",      162),
    ("pl_team",        "all"),
    ("pl_mode",        MODE_SINGLE),
    ("pl_sort_worst",  False),
    ("pl_show_min_ip", False),
    ("pl_show_player_ip", False),
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
    key="pl_stat",
    format_func=lambda x: label_map.get(x, x),
)

col1, col2 = st.columns([.5, 2])

with col1:
    mode = st.radio(
        "Mode",
        options=[MODE_SINGLE, MODE_SPLIT, MODE_MULTI],
        key="pl_mode",
    )

    if mode == MODE_SINGLE:
        st.number_input(
            "Year",
            min_value=1900,
            max_value=current_year,
            key="pl_year",
        )
        start_year = st.session_state["pl_year"]
        end_year   = st.session_state["pl_year"]
    else:
        st.number_input(
            "Start Year",
            min_value=1900,
            max_value=current_year,
            value=st.session_state.get("pl_start_year", 2024),
            key="pl_start_year",
        )
        st.number_input(
            "End Year",
            min_value=1900,
            max_value=current_year,
            value=st.session_state.get("pl_end_year", 2025),
            key="pl_end_year",
        )
        start_year = st.session_state["pl_start_year"]
        end_year   = st.session_state["pl_end_year"]
        if end_year < start_year:
            end_year = start_year

    st.number_input(
        "Min IP",
        min_value=0,
        max_value=5000,
        key="pl_min_ip",
    )

    team_disabled = (mode == MODE_MULTI)
    st.selectbox(
        "Team",
        options=list(TEAM_OPTIONS.keys()),
        format_func=lambda x: TEAM_OPTIONS[x],
        key="pl_team",
        disabled=team_disabled,
        help="Team filter is not available for Multi-Year Span mode." if team_disabled else None,
    )

    st.checkbox("Show worst",       key="pl_sort_worst")
    st.checkbox("Show min IP",      key="pl_show_min_ip")
    st.checkbox("Show player IP",   key="pl_show_player_ip")

# Resolved values
start_year  = int(start_year)
end_year    = int(max(start_year, end_year))
min_ip_val  = int(st.session_state.get("pl_min_ip", 0))
team_val    = "all" if team_disabled else st.session_state.get("pl_team", "all")

split_seasons_active = (mode == MODE_SPLIT)

# ----------------------------
#  Load data
# ----------------------------
if split_seasons_active:
    df = load_split_season_data(start_year, end_year, min_ip_val)
    if team_val != "all" and not df.empty and "Team" in df.columns:
        df = df[df["Team"].apply(
            lambda v: normalize_team_code(str(v).upper().strip(), start_year) == team_val
            if pd.notna(v) else False
        )]
else:
    df = load_filtered_data(start_year, end_year, min_ip_val)
    if team_val != "all" and not df.empty and "Team" in df.columns:
        df = df[df["Team"].apply(
            lambda v: normalize_team_code(str(v).upper().strip(), start_year) == team_val
            if pd.notna(v) else False
        )]

# Slim df
if not df.empty:
    keep_cols = set(STAT_ALLOWLIST) | {
        "Name", "Team", "TeamDisplay", "IDfg", "Age", "Season",
        "mlbam", "MLBID", "key_mlbam", "mlbam_id", "IP",
    }
    df = df[[c for c in df.columns if c in keep_cols]]

# ----------------------------
#  Sort & take top 10
# ----------------------------
if not df.empty and stat in df.columns:
    is_lower_better = stat in lower_better
    show_worst = st.session_state.get("pl_sort_worst", False)
    ascending = (is_lower_better and not show_worst) or (not is_lower_better and show_worst)
    df = df.sort_values(by=stat, ascending=ascending)
    df = df.head(10)
elif not df.empty:
    st.error(f"Column '{stat}' not found. Available columns: {', '.join(df.columns)}")
    df = pd.DataFrame()

if not df.empty and "TeamDisplay" not in df.columns:
    df["TeamDisplay"] = "N/A"

# ----------------------------
#  Build cards
# ----------------------------
cards = []
for _, row in df.iterrows():
    name        = row.get("Name", "")
    team        = row.get("TeamDisplay", "")
    raw_val     = row.get(stat, np.nan)
    transformed = transform_stat_value(stat, raw_val)
    display_val = format_stat(stat, transformed)

    if split_seasons_active and "Season" in row.index and pd.notna(row.get("Season")):
        season_yr = int(row["Season"])
        team = f"{team} ({season_yr})"

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

    ip_val = row.get("IP", np.nan)
    player_ip_display = (
        f'<div class="player-ip">{format_stat("IP", ip_val)} IP</div>'
        if st.session_state.get("pl_show_player_ip", False) and pd.notna(ip_val)
        else ""
    )

    src      = get_headshot_url_from_row(src_row)
    img_html = f'<img src="{html.escape(src)}" alt="{html.escape(str(name))}"/>'
    card_html = f"""
    <div class="player-card">
      {img_html}
      <div class="player-name">{name}</div>
      <div class="player-team">{team}</div>
      <div class="player-stat">{display_val}</div>
      {player_ip_display}
    </div>
    """
    cards.append(card_html)

# ----------------------------
#  Title
# ----------------------------
if mode == MODE_SINGLE:
    span_label = f"{int(start_year)}"
else:
    span_label = f"{int(start_year)}\u2013{int(end_year)}"

title_label = label_map.get(stat, stat)

team_label = ""
if team_val != "all":
    team_label = TEAM_OPTIONS.get(team_val, "")

mode_label = " Single Season" if mode == MODE_SPLIT else ""

title = f"{span_label}{mode_label} {team_label} {title_label} Leaders".strip()
title = re.sub(r"  +", " ", title)

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
        {"".join(cards)}
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
.player-ip {{
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
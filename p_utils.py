from pathlib import Path
import pandas as pd
import numpy as np
import requests
import boto3
import os
from zoneinfo import ZoneInfo

TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}

start_year = 1901

LOCAL_BWAR_FILE = Path(__file__).with_name("warpitchers.txt")

HEADSHOT_BASE_SILO = (
    "https://img.mlbstatic.com/mlb-photos/image/upload"
    "/d_people:generic:headshot:67:current.png"
    "/w_240,q_auto:best,f_auto/v1/people/{mlbam}/headshot/silo/current"
)
HEADSHOT_BASE_67 = (
    "https://img.mlbstatic.com/mlb-photos/image/upload"
    "/d_people:generic:headshot:67:current.png"
    "/w_240,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current"
)
HEADSHOT_PLACEHOLDER = (
    "https://img.mlbstatic.com/mlb-photos/image/upload"
    "/w_240,q_auto:best,f_auto/people/generic/headshot/67/current.png"
)

TEAM_ALIASES = {"ATH": "OAK", "ATH/OAK": "OAK", "OAK/ATH": "OAK"}

STAT_DISPLAY_NAMES = {
    "HardHit%": "Hard Hit%",
}



STAT_ALLOWLIST = [
    "fWAR", "bWAR", "fWAR-bWAR Avg",
    "ERA", "xERA", "xBA", "FIP", "xFIP", "ERA-xERA","vFA", "K%", "BB%", "K-BB%", "IP", 
    "Chase%", "Whiff%", "G", "GS",
    "Barrel%", "HardHit%", "EV", "GB%", "K/9","BB/9","K/BB","HR/9", "BABIP", "LOB%", "HR/FB",
    "SV", "AVG", "WHIP", "ERA-", "FIP-", "SIERA",
     "WPA", "Clutch",
    "SO", "BB", "HBP", "HR", "QS", "CG", "ShO", "ER", "TBF", "fWAR/200", "bWAR/200"
]

STAT_ROUND = {
    # 3 decimal places
    "WHIP": 3, "BABIP": 3, "AVG": 3, "xBA": 3,

    # 2 decimal places
    "ERA": 2, "xERA": 2, "FIP": 2, "xFIP": 2, "ERA-xERA": 2,
    "SIERA": 2, "K/9": 2, "BB/9": 2, "HR/9": 2,"K/BB": 2,
    "WPA": 2, "Clutch": 2,

    # 1 decimal place
    "fWAR": 1, "bWAR": 1, "fWAR-bWAR Avg": 1,
    "fWAR/200": 1, "bWAR/200": 1,
    "EV": 1, "vFA": 1, "IP": 1,
    "K%": 1, "BB%": 1, "K-BB%": 1,
    "Chase%": 1, "Whiff%": 1,
    "Barrel%": 1, "HardHit%": 1, "GB%": 1, "LOB%": 1, "HR/FB":1,

    # 0 decimal places
    "ERA-": 0, "FIP-": 0,
    "G": 0, "GS": 0, "SO": 0, "BB": 0, "HBP": 0, "HR": 0,
    "QS": 0, "CG": 0, "ShO": 0, "ER": 0, "TBF": 0, "SV": 0,
}

STAT_DEFAULTS = {
    "fWAR": 4.0, "bWAR": 4.0, "ERA": 3.00, "xERA": 3.00, "FIP": 3.00, "xFIP": 3.00,
    "fWAR-bWAR Avg": 4.0,
    "WHIP": 1.10, "ERA-": 80.0, "FIP-": 80.0, "SIERA": 3.50,
    "IP": 162.0, "G": 30.0, "GS": 25.0, "W": 12.0, "L": 10.0,
    "SV": 20.0, "SO": 180.0, "BB": 50.0,
    "K/9": 10.0, "BB/9": 2.5, "HR/9": 1.0,
    "K%": 25.0, "BB%": 7.0, "K-BB%": 18.0,
    "Barrel%": 6.0, "HardHit%": 35.0, "EV": 88.0,
    "Chase%": 32.0, "Whiff%": 25.0,
    "GB%": 50.0, "HR/FB": 10.0,
    "BABIP": 0.280, "WPA": 2.0, "Clutch": 1.0,
    "CG": 1.0, "ShO": 1.0,
    "fWAR/200": 5.0,
    "bWAR/200": 5.0,
    "ERA-xERA": 0.5,
    "vFA": 95.0,
    "xBA": .250,
}

EVERY_STAT_PRESET = ["fWAR", "bWAR", "fWAR-bWAR Avg", "W-L", "vFA",
        "ERA", "xERA", "xBA", "FIP", "xFIP", "ERA-xERA","IP", "G", "GS", "SO", "BB", "HBP", "HR", "K/9",
        "BB/9", "HR/9", "BABIP", "LOB%", "HR/FB", "QS", "CG", "ShO",
        "SV", "K%", "BB%", "K-BB%", "BB/9","HR/9","K/BB","AVG", "WHIP", "ERA-", "FIP-",
        "Barrel%", "HardHit%", "EV", "GB/FB", "GB%", "FB%", "SIERA",
        "Chase%", "Whiff%", "WPA", "Clutch","fWAR/200", "bWAR/200"]



SUM_STATS = {
    "G", "GS", "HR", "BB", "SO", "HBP", "QS", "CG", "ShO", "SV", "WPA", "W", "L", "fWAR", "bWAR", "TBF", "ER", "fWAR-bWAR Avg"
}
RATE_STATS = {
    "ERA", "xERA", "xBA", "FIP", "xFIP", "K/9", "BB/9", "HR/9", "BABIP", "LOB%", "HR/FB",
    "K%", "BB%", "K-BB%", "AVG", "WHIP", "Barrel%", "HardHit%", "EV",
    "GB/FB", "GB%", "FB%", "SIERA", "Chase%", "Whiff%", "Clutch",
    "ERA-", "FIP-", "vFA","BB/9","HR/9","K/BB","fWAR/200", "bWAR/200", "ERA-xERA",
    "ER","SO","BB","TBF"
}

PCT_STATS = {
    "K%", "BB%", "K-BB%", "Chase%", "Whiff%", "Barrel%", "HardHit%",
    "GB%", "FB%", "LOB%", "HR/FB",
}

label_map = {
    "EV": "Avg Exit Velo",
    "HardHit%": "Hard Hit%",
}

lower_better = {
    "ERA", "xERA", "FIP", "xFIP", "SIERA", "BB", "HBP", "HR",
    "BB/9", "HR/9", "BABIP", "HR/FB", "BB%", "AVG", "WHIP",
    "ERA-", "FIP-", "Barrel%", "HardHit%", "EV", "HR/9","BB/9","ERA-xERA", "xBA"
}


def normalize_team(team: str) -> str:
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)


def get_team_display(team_value: str) -> str:
    t = str(team_value).strip()
    if t == "- - -":
        return "2+ Teams"
    return normalize_team(t)


def _url_has_real_image(url: str) -> bool:
    try:
        r = requests.head(url, timeout=3)
        return r.status_code == 200 and int(r.headers.get("content-length", 0)) > 10000
    except Exception:
        return False

def get_headshot(row: pd.Series) -> str:
    val = row.get("MLBAMID")
    if val is not None and pd.notna(val):
        mlbam = int(val)
        for url_template in (HEADSHOT_BASE_SILO, HEADSHOT_BASE_67):
            url = url_template.format(mlbam=mlbam)
            if _url_has_real_image(url):
                return url
    return HEADSHOT_PLACEHOLDER


# ─────────────────────────────────────────────
#  IP helpers
# ─────────────────────────────────────────────

def ip_to_outs(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
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
        outs_extra = min(max(int(round(fractional * 3)), 0), 2)
    return innings * 3 + outs_extra


def outs_to_ip(outs: float) -> float:
    if pd.isna(outs):
        return np.nan
    innings = int(float(outs) // 3)
    remainder = int(round(float(outs) % 3))
    return innings + remainder / 10

def resolve_player_id(name: str, start_year: int, end_year: int) -> int | None:
    for year in range(end_year, start_year - 1, -1):
        pid = get_player_id_by_name(name, year)
        if pid is not None:
            return pid
    return None


def get_player_id_by_name(name: str, year: int) -> int | None:
    df = load_final_year(year)
    if df is None or df.empty or "Name" not in df.columns:
        return None
    match = df[df["Name"].str.strip() == name.strip()]
    if match.empty:
        match = df[df["Name"].str.lower().str.strip() == name.lower().strip()]
    if match.empty:
        return None
    ids = match["PlayerId"].dropna()
    return int(ids.iloc[0]) if not ids.empty else None

def aggregate_player_group(df: pd.DataFrame) -> pd.DataFrame:
    # ── Numeric coercion ─────────────────────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        if col not in {"Name", "Team", "Pos", "PlayerId", "MLBAMID", "IP"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── IP → outs (vectorized) ───────────────────────────────────────────────
    df["_outs"] = df["IP"].apply(ip_to_outs)

    # ── Name, MLBAMID from last season ───────────────────────────────────────
    last = df.sort_values("Season").groupby("PlayerId", as_index=False).last()[
        ["PlayerId", "Name", "MLBAMID"]
    ]

    # ── Team ─────────────────────────────────────────────────────────────────
    team_info = (
        df.groupby("PlayerId")["Team"]
        .apply(lambda teams: get_team_display_multiseason(teams.tolist()))
        .reset_index()
    )

    # ── IP: sum outs then convert back ───────────────────────────────────────
    ip_agg = df.groupby("PlayerId", as_index=False)["_outs"].sum()
    ip_agg["IP"] = ip_agg["_outs"].apply(outs_to_ip)
    ip_agg["_ip_innings"] = ip_agg["_outs"] / 3.0

# ── TBF ──────────────────────────────────────────────────────────────────
    if "TBF" not in df.columns or df["TBF"].isna().all():
        tbf_cols = [c for c in ("H", "BB", "HBP") if c in df.columns]
        df["_tbf_est"] = df["_outs"] + df[tbf_cols].sum(axis=1)
        df["TBF"] = df["_tbf_est"]

    # ── Sum stats ────────────────────────────────────────────────────────────
    sum_cols = [c for c in SUM_STATS if c in df.columns and c != "IP"]
    summed = df.groupby("PlayerId", as_index=False)[sum_cols].sum(min_count=1)

    # ── Outs-weighted rate stats ─────────────────────────────────────────────
    rate_cols = [
        c for c in df.columns
        if c not in SUM_STATS
        and c not in {"PlayerId", "MLBAMID", "Season", "Name", "Team", "IP",
                      "_outs", "_tbf_est", "_ip_innings"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    rate_parts = {"PlayerId": df["PlayerId"]}
    for col in rate_cols:
        mask = df[col].notna()
        rate_parts[f"_w_{col}"]  = df[col] * df["_outs"].where(mask, 0)
        rate_parts[f"_ou_{col}"] = df["_outs"].where(mask, 0)

    rate_df = pd.DataFrame(rate_parts).groupby("PlayerId", as_index=False).sum()
    rated = pd.DataFrame({"PlayerId": rate_df["PlayerId"]})
    for col in rate_cols:
        denom = rate_df[f"_ou_{col}"]
        rated[col] = rate_df[f"_w_{col}"] / denom.replace(0, float("nan"))

    # ── Merge everything ─────────────────────────────────────────────────────
    result = last.merge(team_info, on="PlayerId", how="left")
    result = result.merge(ip_agg[["PlayerId", "IP", "_ip_innings"]], on="PlayerId", how="left")
    result = result.merge(summed,  on="PlayerId", how="left")
    result = result.merge(rated,   on="PlayerId", how="left")

    # ── Recalculate derived stats ─────────────────────────────────────────────
    ip  = result["_ip_innings"]
    tbf = pd.to_numeric(result["TBF"], errors="coerce")
    bb  = pd.to_numeric(result.get("BB"),   errors="coerce")
    so  = pd.to_numeric(result.get("SO"),   errors="coerce")
    er  = pd.to_numeric(result.get("ER"),   errors="coerce")
    fwar = pd.to_numeric(result.get("fWAR"), errors="coerce")
    bwar = pd.to_numeric(result.get("bWAR"), errors="coerce")

    if "ERA" in result.columns:
        result["ERA"] = (er / ip.replace(0, float("nan"))) * 9
    if "BB%" in result.columns:
        result["BB%"] = (bb / tbf.replace(0, float("nan"))) * 100
    if "K%" in result.columns:
        result["K%"] = (so / tbf.replace(0, float("nan"))) * 100
    if "BB/9" in result.columns:
        result["BB/9"] = (bb / ip.replace(0, float("nan"))) * 9
    if "K/9" in result.columns:
        result["K/9"] = (so / ip.replace(0, float("nan"))) * 9
    if "fWAR/200" in result.columns:
        result["fWAR/200"] = fwar / ip.replace(0, float("nan")) * 200
    if "bWAR/200" in result.columns:
        result["bWAR/200"] = bwar / ip.replace(0, float("nan")) * 200
    if "fWAR-bWAR Avg" in result.columns:
        result["fWAR-bWAR Avg"] = (fwar + bwar) / 2

    result = result.drop(columns=["_ip_innings"], errors="ignore")

    return result

def aggregate_player_group_single(grp: pd.DataFrame, start_year: int = 2015) -> dict:
    result: dict = {}

    result["Name"] = str(grp["Name"].iloc[0])
    result["PlayerId"] = grp["PlayerId"].iloc[0]
    result["MLBAMID"] = grp["MLBAMID"].iloc[0]

    teams = grp["Team"].astype(str).tolist()
    result["Team"] = get_team_display_multiseason(teams)

    outs_series = pd.to_numeric(grp["IP"], errors="coerce").apply(ip_to_outs)
    ip_outs_total = outs_series.sum(skipna=True)
    result["IP"] = outs_to_ip(ip_outs_total)

    weight = outs_series.fillna(0)
    weight_total = weight.sum()

    if "TBF" in grp.columns and not grp["TBF"].isna().all():
        tbf_total = pd.to_numeric(grp["TBF"], errors="coerce").sum(skipna=True)
    else:
        tbf_total = ip_outs_total + sum(
        pd.to_numeric(grp[c], errors="coerce").sum(skipna=True)
        for c in ("H", "BB", "HBP") if c in grp.columns
    )

    for col in grp.columns:
        if not pd.api.types.is_numeric_dtype(grp[col]) or col in {"PlayerId", "MLBAMID", "Season", "IP"}:
            continue
        series = pd.to_numeric(grp[col], errors="coerce")
        if series.isna().all():
            continue
        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in RATE_STATS: 
            mask = series.notna()
            w_stat = weight.where(mask, 0)
            w_stat_total = w_stat.sum()
            result[col] = (series * w_stat).sum(skipna=True) / w_stat_total if w_stat_total > 0 else float("nan")

    ip_innings = ip_outs_total / 3.0
    bb, so, er, fwar, bwar = (pd.to_numeric(result.get(c), errors="coerce") for c in ("BB", "SO", "ER", "fWAR", "bWAR"))

    if pd.notna(er) and ip_innings > 0:
        result["ERA"] = (er / ip_innings) * 9
    if pd.notna(bb) and tbf_total > 0:
        result["BB%"] = (bb / tbf_total) * 100
    if pd.notna(so) and tbf_total > 0:
        result["K%"] = (so / tbf_total) * 100
    if pd.notna(bb) and ip_innings > 0:
        result["BB/9"] = (bb / ip_innings) * 9
    if pd.notna(so) and ip_innings > 0:
        result["K/9"] = (so / ip_innings) * 9
    if pd.notna(fwar) and ip_innings > 0:
        result["fWAR/200"] = fwar / ip_innings * 200
    if pd.notna(bwar) and ip_innings > 0:
        result["bWAR/200"] = bwar / ip_innings * 200
    result["fWAR-bWAR Avg"] = (fwar + bwar) / 2

    return result

def get_team_display_multiseason(teams: list[str]) -> str:
    if any(get_team_display(t) == "2+ Teams" for t in teams):
        return "2+ Teams"
    normalized = {normalize_team(t) for t in teams if str(t).strip() and str(t).strip() != "- - -"}
    if len(normalized) > 1:
        return "2+ Teams"
    return normalized.pop() if normalized else "N/A"

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FWAR", "BWAR","FWAR/200", "BWAR/200","FWAR-BWAR AVG"}:
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"EV", "vFA"}:
        return f"{float(val):.1f}"

    if upper_stat in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"

    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "SIERA", "K/9", "BB/9", "HR/9", "GB/FB","ERA-XERA"}:
        return f"{float(val):.2f}"

    if upper_stat == "WHIP":
        return f"{float(val):.3f}"

    if upper_stat == "IP":
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"ERA-", "FIP-"}:
        return f"{int(round(float(val)))}"

    if upper_stat in {"BABIP", "AVG", "XBA"}:
        return f"{float(val):.3f}".lstrip("0") or ".000"
    ALREADY_PCT_POINTS = {"Barrel%", "HardHit%"}
    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat or "BB%" in stat or "HR/FB" in stat
        or "Chase" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1 and stat not in ALREADY_PCT_POINTS:
            v *= 100
        return f"{v:.1f}%"

    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def format_stat_yoy(stat: str, val, show_sign: bool = False) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FWAR", "BWAR","FWAR/200", "BWAR/200","FWAR-BWAR AVG"}:
        v = float(val)
        formatted = f"{int(round(abs(v)))}.0" if abs(v - round(v)) < 1e-9 else f"{abs(v):.1f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"EV", "vFA"}:
        v = float(val)
        formatted = f"{abs(v):.1f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"WPA", "CLUTCH"}:
        v = float(val)
        return f"+{v:.2f}" if show_sign and v > 0 else f"{v:.2f}"

    if upper_stat in {"ERA", "FIP", "XFIP", "XERA", "SIERA", "K/9", "BB/9", "HR/9", "GB/FB","ERA-XERA"}:
        v = float(val)
        formatted = f"{abs(v):.2f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat == "WHIP":
        v = float(val)
        formatted = f"{abs(v):.3f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat == "IP":
        v = float(val)
        formatted = f"{int(round(abs(v)))}.0" if abs(v - round(v)) < 1e-9 else f"{abs(v):.1f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"ERA-", "FIP-"}:
        v = int(round(float(val)))
        return f"+{v}" if show_sign and v > 0 else f"{v}"

    if upper_stat in {"BABIP", "AVG","XBA"}:
        v = float(val)
        formatted = f"{abs(v):.3f}".lstrip("0") or ".000"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat or "BB%" in stat or "HR/FB" in stat
        or "Chase" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        formatted = f"{abs(v):.1f}%"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{abs(v):.1f}%" if v < 0 else formatted

    v = float(val)
    formatted = f"{abs(v):.0f}" if abs(v - round(v)) < 1e-6 else f"{abs(v):.1f}"
    if show_sign and v > 0:
        return f"+{formatted}"
    return f"-{formatted}" if v < 0 else formatted

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

bucket = "sports-analytics-files"

def load_final_year(year: int) -> pd.DataFrame:
    key = f"processed/pitching_final_{year}.csv"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj["Body"])
        df["Season"] = year
        return df
    except Exception:
        return pd.DataFrame()
    

# old loading function
    """@st.cache_data(show_spinner=False, ttl=900)
def load_final_year(year: int) -> pd.DataFrame:
    path = f"data/final/pitching_final_{year}.csv"
    try:
        df = pd.read_csv(path)
        df["Season"] = year
    
        return df
    except Exception:
        return pd.DataFrame()"""
    
STAT_PRESETS = {
    "Default": [
        "fWAR", "bWAR", "ERA", "xERA", "FIP", "IP",
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
    "Statcast": [
        "xERA", "xBA", "vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
    "Stat Mix": [
        "ERA", "xERA", "xBA", "FIP", "xFIP", "EV", "Chase%", "Whiff%", "K%", "BB%", "K-BB%","Barrel%", "HardHit%", "GB%","SIERA",
    ],
    "Standard": [
        "fWAR", "bWAR", "ERA", "G", "GS", "IP", "AVG", "WHIP", "HR/9",
    ], 
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": ["fWAR"],
    "Player A leads": [],
    "Player B leads": [],
    "Player C leads": [],
    "Player D leads": [],
    "Player E leads": [],
}

STAT_PRESETS_SAVANT = {
    "Default": [
        "fWAR", "bWAR",  "ERA", "xERA", "FIP", "IP",
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
    "Statcast": [
         "xERA", "xBA", "vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
    "Stat Mix": [
         "ERA", "xERA", "FIP","xFIP",  "xBA", "vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%", "SIERA",
    ],
    "Standard": [
        "fWAR", "bWAR", "ERA", "GS", "IP", "AVG", "WHIP", "HR/9", "K/BB",
    ],
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": ["fWAR"],
}

STAT_PRESETS_YOY = {
    "Default": [
        "ERA", "xERA", "FIP","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
    "Statcast": [
         "xERA", "xBA", "vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
    "Luck": [
         "ERA","xERA", "EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "LOB%", "BABIP",
    ],
    "Stat Mix": [
        "fWAR", "bWAR", "GS", "IP", "ERA",  "FIP", 
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
    "Stat Mix Rate Basis": [
        "fWAR/200", "bWAR/200", "ERA",  "FIP", 
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
    "Standard": [
        "fWAR", "bWAR", "ERA", "G", "GS", "IP", "AVG", "WHIP", "HR/9",
    ], 
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": ["fWAR"],
    "Only Improvements": [],
    "Only Regressions": [],
}

STAT_PRESETS_DATABASE = {
    "Default": [
        "fWAR", "bWAR",  "fWAR-bWAR Avg","GS","IP", "ERA", "xERA", "FIP", "xFIP", 
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
    "Statcast": [
        "ERA","xERA", "xBA", "ERA-xERA","vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
   
    "Standard": [
   "G", "GS", "IP", 
    "K/9","BB/9","K/BB","K-BB%", "HR/9", "BABIP", "LOB%", "HR/FB",
    "SV", "AVG", "WHIP",  "SO", "BB", "HBP", "HR", "QS", "CG", "ShO", "ER", "TBF", 
    ],
    "Misc": [ "fWAR/200", "bWAR/200", "fWAR-bWAR Avg", "ERA-", "FIP-", "SIERA", "WPA", "Clutch", 
    ] ,
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Add your own": ["fWAR"],

}

STAT_PRESETS_RANKS = {
    "Default": [
        "fWAR", "bWAR",  "fWAR-bWAR Avg","GS","IP", "ERA", "xERA", "FIP", "xFIP", 
        "K%", "BB%", "Whiff%", "Chase%", 
    ],
    "Statcast": [
     "xERA", "xBA", "vFA","EV", "Chase%", "Whiff%", "K%", "BB%", "Barrel%", "HardHit%", "GB%",
    ],
     "Stat Mix": [
        "fWAR", "bWAR", "GS", "IP", "ERA",  "FIP", 
        "K%", "BB%", "Whiff%", "Chase%", "HardHit%", "GB%",
    ],
     "Standard": [
        "fWAR", "bWAR", "ERA", "G", "GS", "IP", "AVG", "WHIP", "HR/9",
    ], 
   
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Add your own": ["fWAR"],

}

def get_last_updated(year: int) -> str:
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"processed/pitching_final_{year}.csv")
        last_modified = obj["LastModified"].astimezone(ZoneInfo("America/New_York"))
        return last_modified.strftime("%B %d, %Y")
    except Exception:
        return "unknown"

def load_risers_data(
    start_year: int, end_year: int,
    min_ip_start: int = 0, min_ip_end: int = 0, team: str = "all"
) -> pd.DataFrame:
    df_s = load_final_year(start_year)
    df_e = load_final_year(end_year)

    if df_s is None or df_s.empty or df_e is None or df_e.empty:
        return pd.DataFrame()

    # Min IP filter
    if min_ip_start > 0:
        df_s = df_s[pd.to_numeric(df_s.get("IP", 0), errors="coerce").fillna(0) >= min_ip_start]
    if min_ip_end > 0:
        df_e = df_e[pd.to_numeric(df_e.get("IP", 0), errors="coerce").fillna(0) >= min_ip_end]

    # Team filter on end year only
    if team != "all" and "Team" in df_e.columns:
        target = normalize_team(team)
        df_e = df_e[df_e["Team"].astype(str).apply(normalize_team) == target]

    if "PlayerId" not in df_s.columns or "PlayerId" not in df_e.columns:
        return pd.DataFrame()

    df_s = df_s.set_index("PlayerId")
    df_e = df_e.set_index("PlayerId")
    common_ids = df_s.index.intersection(df_e.index)

    if len(common_ids) == 0:
        return pd.DataFrame()

    df_s = df_s.loc[common_ids]
    df_e = df_e.loc[common_ids]

    skip = {"Season", "Name", "Team", "MLBAMID", "NameASCII"}
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
            "IP_start": pd.to_numeric(row_s.get("IP", np.nan), errors="coerce"),
            "IP_end":   pd.to_numeric(row_e.get("IP", np.nan), errors="coerce"),
        }

        # Carry MLBAMID for headshots
        for col in ["MLBAMID", "mlbamid"]:
            val = row_e.get(col)
            if val is not None and pd.notna(val):
                record[col] = val
                break

        for col in numeric_cols:
            s_val = pd.to_numeric(row_s.get(col, np.nan), errors="coerce")
            e_val = pd.to_numeric(row_e.get(col, np.nan), errors="coerce")
            record[f"{col}_start"] = s_val
            record[f"{col}_end"]   = e_val
            decimal = STAT_ROUND.get(col,1)
            record[col] = e_val.round(decimal) - s_val.round(decimal)

        rows.append(record)

    return pd.DataFrame(rows)
    

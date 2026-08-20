import pandas as pd
import requests
import boto3
import os
import numpy as np
from zoneinfo import ZoneInfo

TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}

start_year = 1901

STAT_ALLOWLIST = [
    "fWAR", "bWAR", "fWAR-bWAR Avg","xwOBA","wOBA",  "wOBA-xwOBA", "xBA", "xSLG",
    "EV", "Barrel%","HardHit%","Sweet-Spot%",
    "BatSpd","Squared-Up%",  "Whiff%", "Chase%",
    "K%", "BB%", "Off", "Def", "BsR", 
    "FRV", "OAA", "DRS",
    "wRC+", "maxEV","OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "TB", "H",
    "1B", "2B", "3B", "SB", "BB", "IBB", "SO", "HBP","SF",
     "BB/K", "WPA", "Clutch",
     "FRM", "TZ","Swing%", "Z-Swing%", "Z-Swing% - Chase%",
    "O-Contact%", "Z-Contact%", "Zone%",  "Inn",
    "fWAR/650","bWAR/650", "DRS/1350", "OAA/1350","FRV/1350","FRM/1350",
]

STAT_ROUND = {
    # 3 decimal places
    "wOBA": 3, "xwOBA": 3, "wOBA-xwOBA": 3, "xBA": 3, "xSLG": 3,
    "OPS": 3, "SLG": 3, "OBP": 3, "AVG": 3, "ISO": 3, "BABIP": 3,

    # 2 decimal places
    "WPA": 2, "Clutch": 2, "BB/K": 2,

    # 1 decimal place
    "fWAR": 1, "bWAR": 1, "fWAR-bWAR Avg": 1,
    "EV": 1, "BatSpd": 1, "maxEV": 1,
    "Off": 1, "Def": 1, "BsR": 1,
    "FRM": 1, "FRM/1350": 1,
    "fWAR/650": 1, "bWAR/650": 1,
    "Barrel%": 1, "HardHit%": 1, "Sweet-Spot%": 1,
    "Squared-Up%": 1, "Chase%": 1, "Whiff%": 1,
    "K%": 1, "BB%": 1, "Swing%": 1, "Z-Swing%": 1,
    "Z-Swing% - Chase%": 1, "O-Contact%": 1, "Z-Contact%": 1, "Zone%": 1,
    "Inn": 1,

    # 0 decimal places (integers)
    "wRC+": 0,
    "FRV": 0, "OAA": 0, "DRS": 0, "TZ": 0,
    "DRS/1350": 0, "OAA/1350": 0, "FRV/1350": 0,
    "G": 0, "PA": 0, "AB": 0, "R": 0, "RBI": 0, "HR": 0,
    "XBH": 0, "TB": 0, "H": 0, "1B": 0, "2B": 0, "3B": 0,
    "SB": 0, "BB": 0, "IBB": 0, "SO": 0,
}

SUM_STATS = {
    "G", "PA", "AB", "R", "H", "1B", "2B", "3B", "HR", "RBI", "SB",
    "BB", "IBB", "SO", "HBP", "SF", "SH", "XBH", "TB",
    "fWAR", "bWAR", "fWAR-bWAR Avg", "Off", "Def", "BsR",
    "DRS", "OAA", "FRV",
    "WPA", "FRM", "TZ", "Inn"
}

RATE_STATS = {
    "AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "BABIP",
    "K%", "BB%", "K-BB%", "O-Swing%", "Whiff%",
    "Barrel%", "HardHit%",
    "EV","BB/K", "ISO", "BatSpd",
    "wRC+", "Clutch", "Chase%", "Swing%", "Z-Swing%",
    "O-Contact%", "Z-Contact%", "Zone%", "wOBA-xwOBA",
    "Squared-Up%", "fWAR/650","bWAR/650","Sweet-Spot%",
    "DRS/1350", "OAA/1350","FRV/1350","FRM/1350", "Z-Swing% - Chase%"
}

PCT_STATS = {
    "K%", "BB%", "Chase%", "Whiff%", "Swing%", "Z-Swing%",
    "O-Contact%", "Z-Contact%", "Zone%", "Barrel%", "HardHit%",
    "Sweet-Spot%", "Squared-Up%", "Z-Swing% - Chase%",
}

MAX_STATS = {"maxEV"}

EVERY_STAT_PRESET = [
    "bWAR", "fWAR", "fWAR-bWAR Avg", "G", "AB", "PA",  "SB", "HR", "RBI", "XBH",
    "AVG", "OBP", "SLG", "OPS", "ISO", "BABIP",
    "wRC+", "Off", "BsR", "Def", "OAA", "FRV", "FRM", "wOBA",
    "xwOBA", "wOBA-xwOBA","xBA", "xSLG", "EV", "maxEV", "Barrel%", "HardHit%",
    "Chase%", "Whiff%", "K%", "BB%", "BB/K","BB", "IBB", "SO",
    "H", "1B", "2B", "3B",  "TB", "R",
    "K-BB%", "DRS", "WPA", "Clutch", "Swing%", "Z-Swing%", "Z-Swing% - Chase%"
    "O-Contact%","Z-Contact%","Zone%","BatSpd", "Squared-Up%", "Inn"
    "fWAR/650","bWAR/650","DRS/1350", "OAA/1350","FRV/1350","FRM/1350",
]

STAT_DEFAULTS = {
    "HR": 30, "SB": 30, "RBI": 100, "R": 100, "H": 150,
    "fWAR": 4.0, "bWAR": 4.0, "wRC+": 130, "wOBA": 0.370, "OPS": 0.900,
    "fWAR-bWAR Avg": 4.0,
    "xwOBA": 0.370, "xBA": 0.280, "xSLG": 0.480,
    "AVG": 0.300, "OBP": 0.370, "SLG": 0.500, "ISO": 0.200,
    "K%": 20.0, "BB%": 10.0, "Barrel%": 12.0, "HardHit%": 45.0,
    "EV": 92.0, "BB": 60, "IBB": 10, "SO": 100, "PA": 502, "AB": 450, "BB/K": 1.0,
    "2B": 30, "1B": 100, "3B": 5, "XBH": 50, "TB": 300, "G": 140,
    "Clutch": 1.0, "FRV": 10, "OAA": 10, "DRS": 10,
    "Chase%": 25.0, "Whiff%": 20.0,
    "Off": 10.0, "Def": 5.0, "BsR": 3.0,
    "BABIP": 0.320, "WPA": 2.0, "wOBA-xwOBA": 0.020,
    "FRM": 5,
    "Swing%": 45.0, "Z-Swing%": 65.0, "O-Contact%": 65.0,
    "Z-Contact%": 85.0, "Zone%": 45.0,
    "maxEV": 112.0, "BatSpd": 73.0,
    "TZ": 10,
    "Squared-Up%": 25.0,
    "fWAR/650": 5.0,
    "bWAR/650": 5.0,
    "Sweet-Spot%": 34.0,
    "DRS/1350":10, 
    "OAA/1350":10,
    "FRV/1350":10,
    "FRM/1350":5.0,
    "Z-Swing% - Chase%": 20.0,
}

STAT_DISPLAY_NAMES = {
    "HardHit%": "Hard Hit%",
    "EV": "Avg EV",
    "BatSpd": "Bat Speed"
}

STATCAST_RATE_STATS = {"xwOBA", "xBA", "xSLG", "EV", "Barrel%", "HardHit%", "Squared-Up%"}

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


label_map = {
    "HardHit%": "Hard Hit%",
    "EV": "Avg EV",
    "BatSpd": "Bat Speed"
}
lower_better = {"K%", "Chase%", "Whiff%","SO"}


TEAM_ALIASES = {
    "ATH": "OAK",
    "ATH/OAK": "OAK",
    "OAK/ATH": "OAK",
}

POSITION_OPTIONS = {
    "all": "All Positions",
    "C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS",
    "LF": "LF", "CF": "CF", "RF": "RF", "OF": "OF", "DH": "DH",
}



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

def normalize_team(team: str) -> str:
    t = str(team).strip()
    return TEAM_ALIASES.get(t, t)

def get_team_display(team_value: str) -> str:
    """
    Simple rule:
      - '- - -' means player was on 2+ teams → '2+ Teams'
      - Otherwise show the (normalized) team abbreviation
    """
    t = str(team_value).strip()
    if t == "- - -":
        return "2+ Teams"
    return normalize_team(t)

def aggregate_player_group(df: pd.DataFrame) -> pd.DataFrame:
    # ── Numeric coercion ─────────────────────────────────────────────────────
    for col in df.select_dtypes(include="object").columns:
        if col not in {"Name", "Team", "Pos", "PlayerId", "MLBAMID"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["PA"]  = pd.to_numeric(df.get("PA"),  errors="coerce").fillna(0)
    df["Inn"] = pd.to_numeric(df.get("Inn"), errors="coerce").fillna(0)

    g = df.groupby("PlayerId", as_index=False)

    # ── Name, MLBAMID from last season ───────────────────────────────────────
    last = df.sort_values("Season").groupby("PlayerId", as_index=False).last()[
        ["PlayerId", "Name", "Pos", "MLBAMID"]
    ]

    # ── Team: 2+ Teams if multiple distinct teams ────────────────────────────
    team_info = (
        df.groupby("PlayerId")["Team"]
        .apply(lambda teams: (
            "2+ Teams"
            if len({normalize_team(t) for t in teams if str(t).strip() not in ("", "- - -")}) > 1
            else normalize_team(str(teams.iloc[0]))
        ))
        .reset_index()
    )

    # ── Sum stats ────────────────────────────────────────────────────────────
    sum_cols = [c for c in SUM_STATS if c in df.columns]
    summed = g[sum_cols].sum(min_count=1)

    # ── Max stats ────────────────────────────────────────────────────────────
    max_cols = [c for c in MAX_STATS if c in df.columns]
    maxed = g[max_cols].max()

    # ── PA-weighted rate stats ───────────────────────────────────────────────
    rate_cols = [
        c for c in df.columns
        if c not in SUM_STATS and c not in MAX_STATS
        and c not in {"PlayerId", "MLBAMID", "Season", "Name", "Team", "Pos"}
        and pd.api.types.is_numeric_dtype(df[c])
        and c not in {"PA", "Inn"}
    ]

    rate_parts = {"PlayerId": df["PlayerId"]}
    for col in rate_cols:
        mask = df[col].notna()
        rate_parts[f"_w_{col}"]  = df[col] * df["PA"].where(mask, 0)
        rate_parts[f"_pa_{col}"] = df["PA"].where(mask, 0)

    rate_df = pd.DataFrame(rate_parts).groupby("PlayerId", as_index=False).sum()

    rated = pd.DataFrame({"PlayerId": rate_df["PlayerId"]})
    for col in rate_cols:
        denom = rate_df[f"_pa_{col}"]
        rated[col] = rate_df[f"_w_{col}"] / denom.replace(0, float("nan"))

    # ── Merge everything ─────────────────────────────────────────────────────
    result = last.merge(team_info, on="PlayerId", how="left")
    result = result.merge(summed,  on="PlayerId", how="left")
    result = result.merge(maxed,   on="PlayerId", how="left")
    result = result.merge(rated,   on="PlayerId", how="left")

    # ── Recalculate derived stats from aggregated counting stats ─────────────
    ab   = pd.to_numeric(result.get("AB"),  errors="coerce")
    h    = pd.to_numeric(result.get("H"),   errors="coerce")
    bb   = pd.to_numeric(result.get("BB"),  errors="coerce").fillna(0)
    hbp  = pd.to_numeric(result.get("HBP"), errors="coerce").fillna(0)
    sf   = pd.to_numeric(result.get("SF"),  errors="coerce").fillna(0)
    tb   = pd.to_numeric(result.get("TB"),  errors="coerce")
    pa   = pd.to_numeric(result.get("PA"),  errors="coerce")
    inn  = pd.to_numeric(result.get("Inn"), errors="coerce")
    fwar = pd.to_numeric(result.get("fWAR"), errors="coerce")
    bwar = pd.to_numeric(result.get("bWAR"), errors="coerce")
    drs  = pd.to_numeric(result.get("DRS"),  errors="coerce")
    oaa  = pd.to_numeric(result.get("OAA"),  errors="coerce")
    frv  = pd.to_numeric(result.get("FRV"),  errors="coerce")
    frm  = pd.to_numeric(result.get("FRM"),  errors="coerce")
    so = pd.to_numeric(result.get("SO"), errors="coerce")

    if "K%" in result.columns:
        result["K%"] = so / pa.replace(0, float("nan"))
    if "BB%" in result.columns:
        result["BB%"] = bb / pa.replace(0, float("nan"))

    if "AVG" in result.columns:
        result["AVG"] = h / ab.replace(0, float("nan"))
    if "SLG" in result.columns:
        result["SLG"] = tb / ab.replace(0, float("nan"))
    if "OBP" in result.columns:
        obp_den = ab.fillna(0) + bb + hbp + sf
        result["OBP"] = (h + bb + hbp) / obp_den.replace(0, float("nan"))
    if "OPS" in result.columns:
        result["OPS"] = result["SLG"] + result["OBP"]
    if "ISO" in result.columns:
        result["ISO"] = result["SLG"] - result["AVG"]
    if "fWAR-bWAR Avg" in result.columns:
        result["fWAR-bWAR Avg"] = (fwar + bwar) / 2
    if "fWAR/650" in result.columns:
        result["fWAR/650"] = fwar / pa.replace(0, float("nan")) * 650
    if "bWAR/650" in result.columns:
        result["bWAR/650"] = bwar / pa.replace(0, float("nan")) * 650
    if "DRS/1350" in result.columns:
        result["DRS/1350"] = drs / inn.replace(0, float("nan")) * 1350
    if "OAA/1350" in result.columns:
        result["OAA/1350"] = oaa / inn.replace(0, float("nan")) * 1350
    if "FRV/1350" in result.columns:
        result["FRV/1350"] = frv / inn.replace(0, float("nan")) * 1350
    if "FRM/1350" in result.columns:
        result["FRM/1350"] = frm / inn.replace(0, float("nan")) * 1350

    return result

def aggregate_player_group_single(grp: pd.DataFrame) -> dict:
    result: dict = {}

    result["Name"] = str(grp["Name"].iloc[0])
    result["PlayerId"] = grp["PlayerId"].iloc[0]
    result["MLBAMID"] = grp["MLBAMID"].iloc[0]

    teams = grp["Team"].astype(str).tolist()
    result["Team"] = (
        "2+ Teams" if any(get_team_display(t) == "2+ Teams" for t in teams)
        or len({normalize_team(t) for t in teams if t.strip() and t.strip() != "- - -"}) > 1
        else normalize_team(teams[0]) if teams else "N/A"
    )

    pa_weight = pd.to_numeric(grp["PA"], errors="coerce").fillna(0)
    inn_weight = pd.to_numeric(grp["Inn"], errors="coerce").fillna(0)
    pa_total = pa_weight.sum()
    inn_total = inn_weight.sum()

    for col in grp.columns:
        if not pd.api.types.is_numeric_dtype(grp[col]) or col in {"PlayerId", "MLBAMID", "Season"}:
            continue
        series = pd.to_numeric(grp[col], errors="coerce")
        if series.isna().all():
            continue
        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in MAX_STATS:
            result[col] = series.max(skipna=True)
        else:
            mask = series.notna()
            pa_stat = pa_weight.where(mask, 0)
            pa_stat_total = pa_stat.sum()
            result[col] = (series * pa_stat).sum(skipna=True) / pa_stat_total if pa_stat_total > 0 else float("nan")

    h, ab, bb, hbp, sf, tb, fwar, bwar, drs, oaa, frv, frm, so = (
        pd.to_numeric(result.get(c), errors="coerce")
        for c in ("H", "AB", "BB", "HBP", "SF", "TB", "fWAR", "bWAR", "DRS", "OAA", "FRV", "FRM","SO")
    )

    if pd.notna(ab) and ab > 0 and pd.notna(h):
        result["AVG"] = h / ab
    if pd.notna(ab) and ab > 0 and pd.notna(tb):
        result["SLG"] = tb / ab
    if pd.notna(pa_total) and pa_total > 0 and pd.notna(so):
        result["K%"] = so / pa_total
    if pd.notna(pa_total) and pa_total > 0 and pd.notna(bb):
        result["BB%"] = bb / pa_total
    if pd.notna(pa_total) and pa_total > 0 and pd.notna(fwar):
        result["fWAR/650"] = fwar / pa_total * 650
    if pd.notna(pa_total) and pa_total > 0 and pd.notna(bwar):
        result["bWAR/650"] = bwar / pa_total * 650
    if pd.notna(inn_total) and inn_total > 0 and pd.notna(drs):
        result["DRS/1350"] = drs / inn_total * 1350
    if pd.notna(inn_total) and inn_total > 0 and pd.notna(oaa):
        result["OAA/1350"] = oaa / inn_total * 1350
    if pd.notna(inn_total) and inn_total > 0 and pd.notna(frv):
        result["FRV/1350"] = frv / inn_total * 1350
    if pd.notna(inn_total) and inn_total > 0 and pd.notna(frm):
        result["FRM/1350"] = frm / inn_total * 1350
    result["fWAR-bWAR Avg"] = (fwar + bwar) / 2 if pd.notna(fwar) and pd.notna(bwar) else float("nan")

    bb_v, hbp_v, sf_v = (0 if pd.isna(v) else v for v in (bb, hbp, sf))
    obp_den = (ab if pd.notna(ab) else 0) + bb_v + hbp_v + sf_v
    if obp_den > 0 and pd.notna(h):
        result["OBP"] = (h + bb_v + hbp_v) / obp_den

    slg, obp, avg = (result.get(c) for c in ("SLG", "OBP", "AVG"))
    if pd.notna(slg) and pd.notna(obp):
        result["OPS"] = slg + obp
    if pd.notna(slg) and pd.notna(avg):
        result["ISO"] = slg - avg

    return result

def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""
    upper_stat = stat.upper()

    if upper_stat in {"FRV", "OAA", "DRS","TZ","DRS/1350", "OAA/1350","FRV/1350",}:
        return f"{int(round(float(val)))}"

    if upper_stat in {"Inn","WAR", "BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR", "MAXEV", "BATSPD","FRM", "FWAR/650", "BWAR/650","SWEET-SPOT%","FRM/1350","FWAR-BWAR AVG"}:
        v = float(val)
        return f"{int(round(v))}.0" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

    if upper_stat in {"WPA", "CLUTCH","BB/K"}:
        return f"{float(val):.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO", "WOBA-XWOBA"}:
        return f"{float(val):.3f}".lstrip("0") or ".000"

    if upper_stat in {"WRC+"}:
        return f"{int(round(float(val)))}"
    ALREADY_PCT_POINTS = {"Barrel%", "HardHit%", "Sweet-Spot%"}
    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
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

    if upper_stat in {"FRV", "OAA", "DRS","TZ","DRS/1350", "OAA/1350","FRV/1350",}:
        v = int(round(float(val)))
        return f"+{v}" if show_sign and v > 0 else f"{v}"

    if upper_stat in {"Inn","BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR", "MAXEV", "BATSPD", "FRM", "FWAR/650", "BWAR/650","SWEET-SPOT%","FRM/1350","FWAR-BWAR AVG"}:
        v = float(val)
        formatted = f"{int(round(abs(v)))}.0" if abs(v - round(v)) < 1e-9 else f"{abs(v):.1f}"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"WPA", "CLUTCH","BB/K"}:
        v = float(val)
        return f"+{v:.2f}" if show_sign and v > 0 else f"{v:.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO", "WOBA-XWOBA"}:
        v = float(val)
        formatted = f"{abs(v):.3f}".lstrip("0") or ".000"
        if show_sign and v > 0:
            return f"+{formatted}"
        return f"-{formatted}" if v < 0 else formatted

    if upper_stat in {"WRC+"}:
        v = int(round(float(val)))
        return f"+{v}" if show_sign and v > 0 else f"{v}"

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
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

def apply_dh_override(df):
    if "Pos" not in df.columns or "PA" not in df.columns or "Inn" not in df.columns or "Season" not in df.columns:
        print(df.columns)
        return df

    df = df.copy()


    # row-level DH eligibility
    eligible = df["Season"] >= 1973



    pa = pd.to_numeric(df["PA"], errors="coerce").fillna(0)
    inn = pd.to_numeric(df["Inn"], errors="coerce").fillna(0)

    estimated = (pa / 4.1) * 9

    is_dh = eligible & ((inn == 0) | ((inn > 0) & (estimated / inn > 3)))

    df.loc[is_dh, "Pos"] = "DH"
    return df



def filter_by_position(df, position):
    df["Pos"] = df["Pos"].astype(str).str.strip().str.upper()
    df = apply_dh_override(df)
    
    if position == "all" or "Pos" not in df.columns:
        return df
    
    position = position.upper()
    
    def player_matches(player_df):
        if position == "OF":
            of_positions = {"LF", "CF", "RF"}
            primary = player_df["Pos"].mode().iloc[0]
            return primary in of_positions
        else:
            primary = player_df["Pos"].mode().iloc[0]
            return primary == position
    
    # Group by player, check if their primary pos matches, return all their rows if so
    matched_players = (
        df.groupby("PlayerId")
        .filter(player_matches)
    )
    
    return matched_players


s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

bucket = "sports-analytics-files"


def load_final_year(year: int) -> pd.DataFrame:
    key = f"processed/hitting_final_{year}.csv"
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
    path = f"data/final/hitting_final_{year}.csv"
    try:
        df = pd.read_csv(path)
        df["Season"] = year
    
        return df
    except Exception:
        return pd.DataFrame()"""

def resolve_player_id(name: str, start_year: int, end_year: int) -> int | None:
    """Try each year in range to find the PlayerId for a name."""
    for year in range(end_year, start_year - 1, -1):
        pid = get_player_id_by_name(name, year)
        if pid is not None:
            return pid
    return None

def get_player_id_by_name(name: str, year: int) -> int | None:
    """Look up PlayerId for an exact name match in a given year."""
    df = load_final_year(year)
    if df is None or df.empty or "Name" not in df.columns:
        return None
    # Try exact match first
    match = df[df["Name"].str.strip() == name.strip()]
    if match.empty:
        # Try case-insensitive
        match = df[df["Name"].str.lower().str.strip() == name.lower().strip()]
    if match.empty:
        return None
    ids = match["PlayerId"].dropna()
    return int(ids.iloc[0]) if not ids.empty else None

STAT_PRESETS = {
    "Default": [
        "fWAR", "bWAR", "G", "PA", "HR", "OPS", "wRC+",
        "K%", "BB%",  "BsR", "SB", "FRV", "DRS",
    ],
   
    "Statcast": [
        "xwOBA", "xBA", "xSLG", "EV", "Barrel%", "HardHit%",  "Sweet-Spot%",
        "BatSpd","Squared-Up%",
        "Chase%", "Whiff%", "K%", "BB%",
    ],
    "Standard": [
        "fWAR", "bWAR", "G", "PA", "AVG", "OBP", "SLG", "OPS",
        "H",  "2B", "3B", "HR", "XBH", "RBI", "SB", "R",
    ],
    "Fielding": [
        "DRS", "FRV", "OAA", "Def"
    ],
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": [
        "fWAR",
    ],
    "Player A leads": [],
    "Player B leads": [],
    "Player C leads": [],
    "Player D leads": [],
    "Player E leads": [],
}

STAT_PRESETS_SAVANT = {
     "Statcast": [
        "xwOBA", "xBA", "xSLG", "EV", "Barrel%", "HardHit%", "Sweet-Spot%", "BatSpd", "Squared-Up%", 
        "Chase%", "Whiff%", "K%", "BB%",
    ],
    "Stat Mix": [
        "fWAR", "bWAR", "HR", "OPS", "wRC+",
        "K%", "BB%",  "BsR", "SB", "OAA", "FRV", "DRS",
    ],
    "Fielding": ["DRS", "FRV", "OAA", "Def"],
    
    "Standard": [
        "bWAR", "fWAR", "PA", "AVG", "OBP", "SLG", "OPS",
        "2B", "3B", "HR", "XBH", "RBI", "SB"
    ],
"Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": [
        "fWAR",
    ],
}

STAT_PRESETS_YOY = {
    "Statcast": [
        "xwOBA", "xBA", "xSLG", "EV", "Barrel%", "HardHit%", "Sweet-Spot%",
        "BatSpd","Squared-Up%",
        "Chase%", "Whiff%", "K%", "BB%",
    ],
   "Stat Mix": [
        "fWAR", "bWAR", "G", "PA", "HR", "OPS", "wRC+",
        "K%", "BB%",  "BsR", "SB", "FRV", "DRS",
    ],
    "Standard": [
        "fWAR", "bWAR", "PA", "AVG", "OBP", "SLG", "OPS",
         "HR", "XBH", "RBI", "SB", "R",
        "K%", "BB%", "DRS",
    ],
    "Fielding": [
        "DRS", "FRV", "OAA", "Def"
    ],
    "Fielding Rates": ["DRS/1350", "FRV/1350", "OAA/1350"],
   
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Create your own": [
        "fWAR",
    ],
    "Only Improvements": [],
    "Only Regressions": [],
}

STAT_PRESETS_DATABASE = {
    "Default": [
         "fWAR", "bWAR", "fWAR-bWAR Avg", "G", "PA", "HR", "wRC+", "xwOBA",
        "K%", "BB%", "BsR", "SB", "OAA", "FRV", "DRS",
    ],
   
    "Statcast": [
        "xwOBA", "xBA", "xSLG", "EV", "Barrel%", "HardHit%", "Sweet-Spot%",
        "BatSpd","Squared-Up%",
        "Chase%", "Whiff%", "K%", "BB%",
    ],
    "Standard": [
        "fWAR", "bWAR", "AVG", "OBP", "SLG", "OPS",
        "H", "2B", "3B", "HR", "XBH", "RBI", "SB", "R",
       
    ],
    "Fielding": [
        "DRS", "FRV", "OAA", "Def", "FRM",
    ],
    "Discipline": ["Chase%","Z-Swing%", "Z-Swing% - Chase%", "Swing%", 
    "O-Contact%", "Z-Contact%", "Zone%",],
    "Every Stat": EVERY_STAT_PRESET,
    "Blank – Add your own": [
        "fWAR",
    ],

}

STAT_PRESETS_RANKS = {
    "Default": [
         "fWAR", "bWAR", "fWAR-bWAR Avg", "HR", "OPS", "wRC+", 
        "K%", "BB%", "BsR", "SB", "OAA", "FRV", "DRS",
    ],
    "Statcast": [
        "xwOBA", "xBA", "xSLG", "EV", "Barrel%", "HardHit%", "Sweet-Spot%",
        "BatSpd","Squared-Up%",
        "Chase%", "Whiff%", "K%", "BB%",
    ],
    "Standard": [
        "fWAR", "bWAR", "AVG", "OBP", "SLG", "OPS",
        "H",  "2B", "3B", "HR", "XBH", "RBI", "SB", "R",
       
    ],
    "Fielding": [
        "DRS", "FRV", "OAA", "Def",
    ],
    "Discipline": ["Chase%","Z-Swing%", "Z-Swing% - Chase%", "Swing%", 
    "O-Contact%", "Z-Contact%", "Zone%",],

    "Blank – Add your own": [
        "fWAR",
    ],

}

def get_last_updated(year: int) -> str:
    try:
        obj = s3.get_object(Bucket=bucket, Key=f"processed/hitting_final_{year}.csv")
        last_modified = obj["LastModified"].astimezone(ZoneInfo("America/New_York"))
        return last_modified.strftime("%B %d, %Y")
    except Exception as e:
        return "unknown"
    
def load_risers_data(
    start_year: int, end_year: int,
    min_pa_start: int = 0, min_pa_end: int = 0,
    min_inn_start: int = 0, min_inn_end: int = 0,
    use_inn: bool = False,
    position: str = "all", team: str = "all"
) -> pd.DataFrame:
    df_s = load_final_year(start_year)
    df_e = load_final_year(end_year)

    if df_s is None or df_s.empty or df_e is None or df_e.empty:
        return pd.DataFrame()

    if use_inn:
        if min_inn_start > 0 and "Inn" in df_s.columns:
            df_s = df_s[pd.to_numeric(df_s["Inn"], errors="coerce").fillna(0) >= min_inn_start]
        if min_inn_end > 0 and "Inn" in df_e.columns:
            df_e = df_e[pd.to_numeric(df_e["Inn"], errors="coerce").fillna(0) >= min_inn_end]
    else:
        if min_pa_start > 0:
            df_s = df_s[pd.to_numeric(df_s.get("PA", 0), errors="coerce").fillna(0) >= min_pa_start]
        if min_pa_end > 0:
            df_e = df_e[pd.to_numeric(df_e.get("PA", 0), errors="coerce").fillna(0) >= min_pa_end]

    df_s = filter_by_position(df_s, position)
    df_e = filter_by_position(df_e, position)

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

    skip = {"Season", "Name", "Team", "MLBAMID", "NameASCII", "Pos"}
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
            "PA_start": pd.to_numeric(row_s.get("PA", np.nan), errors="coerce"),
            "PA_end":   pd.to_numeric(row_e.get("PA", np.nan), errors="coerce"),
            "Inn_start": pd.to_numeric(row_s.get("Inn", np.nan), errors="coerce"),
            "Inn_end":   pd.to_numeric(row_e.get("Inn", np.nan), errors="coerce"),
        }

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

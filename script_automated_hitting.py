import pandas as pd
from h_utils import STAT_ALLOWLIST
from io import StringIO
import requests
import time
from utils import BREF_COOKIE, FG_COOKIE, strip_html, _browser_headers, s3, bucket
import numpy as np

localUpload = False

upload = True
 
startYear = 2026
EndYear = 2026

MULTIPLY_100_IF_DECIMAL = {
    "Whiff%", "Chase%", "K%", "BB%", "Swing%", "Z-Swing%",
    "Z-Swing% - Chase%", "O-Contact%", "Z-Contact%", "Zone%",
    "Sweet-Spot%", "Squared-Up%",
}

def fetch_bwar(year: int) -> pd.DataFrame:
    url = "https://www.baseball-reference.com/data/war_daily_bat.txt"
    r = requests.get(url, headers=_browser_headers(BREF_COOKIE))
    r.raise_for_status()
    r.encoding = "utf-8"
    df = pd.read_csv(StringIO(r.text))

    df["MLBAMID"]  = pd.to_numeric(df.get("mlb_ID"),  errors="coerce")
    df["year_ID"]  = pd.to_numeric(df.get("year_ID"), errors="coerce")
    df["bWAR_val"] = pd.to_numeric(df.get("WAR"),     errors="coerce")
    df["Name"] = df.get("name_common")

    df = df.dropna(subset=["MLBAMID", "year_ID", "bWAR_val"])
    df = df[df["year_ID"] == year]

    return df.groupby(["MLBAMID", "year_ID","Name"], as_index=False)["bWAR_val"].sum()

def fetch_statcast_ev(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?csv=true&type=batter&year={year}&position=&team=&min=1"
        "&sort=barrels_per_pa&sortDir=desc"
        f"&_={int(time.time())}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    with requests.Session() as session:
        r = session.get(url, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df[
        ["player_id", "anglesweetspotpercent", "avg_hit_speed", "max_hit_speed", "ev95percent", "brl_percent"]
    ].rename(columns={
        "avg_hit_speed":  "EV",
        "max_hit_speed":  "maxEV",
        "ev95percent":    "HardHit%",
        "brl_percent":    "Barrel%",
        "anglesweetspotpercent": "Sweet-Spot%",
    })
    df["player_id"]           = pd.to_numeric(df["player_id"],           errors="coerce")
    df["EV"]                  = pd.to_numeric(df["EV"],                  errors="coerce")
    df["maxEV"]               = pd.to_numeric(df["maxEV"],               errors="coerce")
    df["HardHit%"]            = pd.to_numeric(df["HardHit%"],            errors="coerce")
    df["Barrel%"]             = pd.to_numeric(df["Barrel%"],             errors="coerce")
    df["Sweet-Spot%"] = pd.to_numeric(df["Sweet-Spot%"], errors="coerce")
    return df


def fetch_expected_stats(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?csv=true&type=batter&year={year}&position=&team=&filterType=bip&min=1"
        f"&_={int(time.time())}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    with requests.Session() as session:
        r = session.get(url, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df[["player_id", "est_ba", "est_slg", "est_woba", "woba"]].rename(columns={
        "est_ba": "xBA", "est_slg": "xSLG", "est_woba": "xwOBA", "woba": "wOBA",
    })
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    for col in ["xBA", "xSLG", "xwOBA", "wOBA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fetch_bat_tracking(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
        f"?csv=true&type=batter&gameType=Regular&minSwings=1&minGroupSwings=1"
        f"&seasonStart={year}&seasonEnd={year}"
        f"&_={int(time.time())}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    with requests.Session() as session:
        r = session.get(url, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df[["id", "avg_bat_speed", "squared_up_per_swing"]].rename(columns={
        "avg_bat_speed": "BatSpd", "squared_up_per_swing": "Squared-Up%",
    })
    df = df.rename(columns={"id": "player_id"})
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    for col in ["BatSpd", "Squared-Up%"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fetch_fangraphs_batting(year: int) -> pd.DataFrame:
    url = (
        "https://www.fangraphs.com/api/leaders/major-league/data"
        f"?age=&pos=all&stats=bat&lg=all&qual=0"
        f"&season={year}&season1={year}"
        f"&startdate={year}-03-01&enddate={year}-11-01"
        f"&month=0&hand=&team=0&pageitems=10000&pagenum=1"
        f"&ind=0&rost=0&players=&postseason=&sortdir=default&sortstat=WAR&download=1"
    )
    r = requests.get(url, headers=_browser_headers(FG_COOKIE))
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["Name"] = df["Name"].apply(strip_html)
    df["Team"] = df["Team"].apply(strip_html)
    df = df.drop(columns=["O-Swing%", "Z-Swing%", "Swing%", "O-Contact%", "Z-Contact%", "Contact%", "Zone%"])
    df = df.rename(columns={
        "WAR":             "fWAR",
        "BaseRunning":     "BsR",
        "xAVG":            "xBA",
        "AvgBatSpeed":     "BatSpd",
        "SquaredUpSwing%": "SqUpSw%",
        "playerid":        "PlayerId",
        "xMLBAMID":        "MLBAMID",
        "scO-Swing%":      "Chase%",
        "scO-Contact%":    "O-Contact%",
        "scO-Zone%":       "Zone%",
        "scZ-Swing%":      "Z-Swing%",
        "scZ-Contact%":    "Z-Contact%",
        "scZ-Zone%":       "Z-Zone%",
        "piSwing%":        "Swing%",
        "piContact%":      "Whiff%",
        "PlayerName":      "NameASCII",
        "Defense":         "Def",
        "Offense":         "Off",
    })
    return df[[
        "Name", "Team", "G", "PA", "HR", "R", "RBI", "SB",
        "BB%", "K%", "ISO", "BABIP", "AVG", "OBP", "SLG",
        "wOBA", "xwOBA", "wRC+", "BsR", "Off", "Def", "fWAR",
        "OPS", "BB", "IBB", "SO", "HBP", "AB", "H", "1B", "2B", "3B", "SF",
        "BB/K", "WPA", "Clutch",
        "EV", "Barrel%", "maxEV", "HardHit%", "xBA", "xSLG",
        "BatSpd", "SqUpSw%",
        "Chase%", "Z-Swing%", "Swing%",
        "O-Contact%", "Z-Contact%", "Whiff%", "Zone%",
        "NameASCII", "PlayerId", "MLBAMID",
    ]]

def fetch_frv(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/fielding-run-value"
        f"?csv=true&seasonStart={year}&seasonEnd={year}"
        "&type=fielder&position=0&minInnings=0.1&minResults=1"
        f"&_={int(time.time())}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    with requests.Session() as session:
        r = session.get(url, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df.rename(columns={"id": "MLBAMID", "total_runs": "FRV"})
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
    df["FRV"] = pd.to_numeric(df["FRV"], errors="coerce")
    return df.groupby("MLBAMID", as_index=False)["FRV"].sum()

def fetch_oaa(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
        f"?type=Fielder&startYear={year}&endYear={year}"
        "&split=yes&team=&range=year&min=1&pos=&roles=&viz=hide&csv=true"
        f"&_={int(time.time())}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    with requests.Session() as session:
        r = session.get(url, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    # OAA CSV uses "mlb_id" and "outs_above_average"; adjust if different
    df = df.rename(columns={"player_id": "MLBAMID", "outs_above_average": "OAA"})
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
    df["OAA"] = pd.to_numeric(df["OAA"], errors="coerce")
    return df.groupby("MLBAMID", as_index=False)["OAA"].sum()

def fetch_fielding(year: int) -> pd.DataFrame:
    EMPTY = pd.DataFrame(columns=["PlayerId", "DRS", "FRM", "Pos", "Inn"])

    url = (
        "https://www.fangraphs.com/api/leaders/major-league/data"
        "?age=&pos=all&stats=fld&lg=all&qual=0"
        f"&season={year}&season1={year}"
        "&startdate=&enddate=&month=0&hand=&team=0"
        "&pageitems=10000&pagenum=1"
        "&ind=0&rost=0&players="
        "&sortdir=default&sortstat=Defense"
    )

    try:
        r = requests.get(url, headers=_browser_headers(FG_COOKIE))
        r.raise_for_status()
        df = pd.DataFrame(r.json()["data"])
    except Exception as e:
        print(f"[fielding] fetch failed: {e}")
        return EMPTY

    if df.empty or "playerid" not in df.columns:
        return EMPTY

    df["PlayerId"]  = pd.to_numeric(df["playerid"],   errors="coerce")
    df["DRS"]       = pd.to_numeric(df.get("DRS"),     errors="coerce")
    df["FRM"]       = pd.to_numeric(df.get("CFraming"), errors="coerce")
    df["Inn"]       = pd.to_numeric(df.get("Inn"),     errors="coerce")

    # Sum DRS, FRM, Inn across all positions per player
    agg = df.groupby("PlayerId", as_index=False)[["DRS", "FRM", "Inn"]].sum()

    # Primary position = position with most innings
    valid = df.dropna(subset=["Inn"])

    pos_by_inn = valid.loc[
        valid.groupby("PlayerId")["Inn"].idxmax(),
        ["PlayerId", "Pos"]
    ].drop_duplicates("PlayerId")

    pos_fallback = df.groupby("PlayerId", as_index=False)["Pos"].first()

    pos = pos_fallback.merge(
        pos_by_inn,
        on="PlayerId",
        how="left",
        suffixes=("_fallback", "")
    )

    pos["Pos"] = pos["Pos"].fillna(pos["Pos_fallback"])
    pos = pos[["PlayerId", "Pos"]]

    return agg.merge(pos, on="PlayerId", how="left")

for YEAR in range(startYear, EndYear + 1):
    
    year_bwar   = fetch_bwar(YEAR)
    fielding_fg = fetch_fielding(YEAR)

    if YEAR >= 2015:
        sv_df          = fetch_statcast_ev(YEAR)
        xw_df          = fetch_expected_stats(YEAR)
        battracking_df = fetch_bat_tracking(YEAR)
        fielding_frv   = fetch_frv(YEAR)
        fielding_oaa   = fetch_oaa(YEAR)
    else:
        sv_df          = pd.DataFrame(columns=["player_id", "EV", "maxEV", "HardHit%", "Barrel%", "Sweet-Spot%"])
        xw_df          = pd.DataFrame(columns=["player_id", "xBA", "xSLG", "xwOBA", "wOBA"])
        battracking_df = pd.DataFrame(columns=["player_id", "BatSpd", "Squared-Up%"])
        fielding_frv   = pd.DataFrame(columns=["MLBAMID", "FRV"])
        fielding_oaa   = pd.DataFrame(columns=["MLBAMID", "OAA"])

    # --- Base: FanGraphs batting ---
    final = fetch_fangraphs_batting(YEAR)
    final.columns  = final.columns.str.strip().str.replace("\ufeff", "")
    final["Year"]  = YEAR
    final["MLBAMID"] = pd.to_numeric(final["MLBAMID"], errors="coerce")

    # --- Merge fielding (drop cols that already exist in final to avoid dupes) ---
    fielding_fg.drop(
        columns=[c for c in fielding_fg.columns if c != "PlayerId" and c in final.columns],
        inplace=True,
    )
    final = final.merge(fielding_fg,  on="PlayerId", how="left")
    final = final.merge(fielding_frv, on="MLBAMID",  how="left")
    final = final.merge(fielding_oaa, on="MLBAMID",  how="left")

    # --- Drop FG cols superseded by Savant versions ---
    final.drop(
        columns=["wOBA", "xwOBA", "xBA", "xSLG", "EV", "maxEV", "HardHit%", "Barrel%", "BatSpd", "SqUpSw%"],
        errors="ignore",
        inplace=True,
    )

    # --- Merge Savant statcast + expected stats ---
    savant_df = sv_df.merge(xw_df, on="player_id", how="outer")
    savant_df["player_id"] = pd.to_numeric(savant_df["player_id"], errors="coerce")
    final = final.merge(savant_df,      left_on="MLBAMID", right_on="player_id", how="left")
    final = final.merge(battracking_df, left_on="MLBAMID", right_on="player_id", how="left")

    # --- Merge bWAR ---
    final = final.merge(year_bwar[["MLBAMID", "bWAR_val","Name"]], on="MLBAMID", how="left")
    final["Name"] = final["Name_y"].fillna(final["Name_x"])
    final.rename(columns={"bWAR_val": "bWAR"}, inplace=True)
    final["bWAR"] = final["bWAR"].fillna(0)

    # --- Derived stats ---
    final["TB"]                = final["1B"] + final["2B"] * 2 + final["3B"] * 3 + final["HR"] * 4
    final["XBH"]               = final["2B"] + final["3B"] + final["HR"]
    final["fWAR-bWAR Avg"]     = (final["fWAR"] + final["bWAR"]) / 2
    final["fWAR/650"]          = final["fWAR"] / final["PA"] * 650
    final["bWAR/650"]          = final["bWAR"] / final["PA"] * 650
    final["wOBA-xwOBA"]        = final["wOBA"] - final["xwOBA"]
    final["Whiff%"]            = 1 - final["Whiff%"]
    final["Z-Swing% - Chase%"] = final["Z-Swing%"] - final["Chase%"]
    final["OAA"] = pd.to_numeric(final["OAA"], errors="coerce")
    final["FRV"] = pd.to_numeric(final["FRV"], errors="coerce")
    final["DRS"] = pd.to_numeric(final["DRS"], errors="coerce")
    final["FRM"] = pd.to_numeric(final["FRM"], errors="coerce")
    final["Inn"] = pd.to_numeric(final["Inn"], errors="coerce")
    inn = final["Inn"].replace(0, np.nan)

    final["DRS/1350"] = (final["DRS"] / inn * 1350).round(0)
    final["OAA/1350"] = (final["OAA"] / inn * 1350).round(0)
    final["FRV/1350"] = (final["FRV"] / inn * 1350).round(0)
    final["FRM/1350"] = (final["FRM"] / inn * 1350).round(1)

    for col in MULTIPLY_100_IF_DECIMAL:
        if col in final.columns:
            numeric = pd.to_numeric(final[col], errors="coerce")
            if numeric.median() <= 1:
                final[col] = numeric * 100

    # --- Ensure every allowlisted col is present ---
    for col in STAT_ALLOWLIST:
        if col not in final.columns:
            final[col] = None


    final = final[["Name", "PlayerId", "MLBAMID", "Pos", "Team"] + list(STAT_ALLOWLIST)]

    # --- Output ---
    if localUpload:
        final.to_csv(f"data/final/hitting_final_{YEAR}.csv", index=False)
        print(f"[{YEAR}] Locally saved {len(final)} players → data/final/hitting_final_{YEAR}.csv")

    if upload:
        csv_buffer = StringIO()
        final.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket,
            Key=f"processed/hitting_final_{YEAR}.csv",
            Body=csv_buffer.getvalue().encode("utf-8"),
        )
        print(f"[{YEAR}] Uploaded {len(final)} players → s3://{bucket}/processed/hitting_final_{YEAR}.csv")

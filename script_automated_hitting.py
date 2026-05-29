import pandas as pd
from pathlib import Path
import boto3
import os
from h_utils import STAT_ALLOWLIST
from io import StringIO
import requests
import re
from utils import BREF_COOKIE, FG_COOKIE, strip_html, _browser_headers

localUpload = True

upload = False
 
startYear = 2026
EndYear = 2026


def fetch_bwar(year: int) -> pd.DataFrame:
    url = "https://www.baseball-reference.com/data/war_daily_bat.txt"
    r = requests.get(url, headers=_browser_headers(BREF_COOKIE))
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))

    df["MLBAMID"]  = pd.to_numeric(df.get("mlb_ID"),  errors="coerce")
    df["year_ID"]  = pd.to_numeric(df.get("year_ID"), errors="coerce")
    df["bWAR_val"] = pd.to_numeric(df.get("WAR"),     errors="coerce")

    df = df.dropna(subset=["MLBAMID", "year_ID", "bWAR_val"])
    df = df[df["year_ID"] == year]

    return df.groupby(["MLBAMID", "year_ID"], as_index=False)["bWAR_val"].sum()

def fetch_statcast_ev(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?csv=true&type=batter&year={year}&position=&team=&min=1"
        "&sort=barrels_per_pa&sortDir=desc"
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
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
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
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
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
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
        "OPS", "BB", "IBB", "SO", "HBP", "AB", "H", "1B", "2B", "3B",
        "BB/K", "WPA", "Clutch",
        "EV", "Barrel%", "maxEV", "HardHit%", "xBA", "xSLG",
        "BatSpd", "SqUpSw%",
        "Chase%", "Z-Swing%", "Swing%",
        "O-Contact%", "Z-Contact%", "Whiff%", "Zone%",
        "NameASCII", "PlayerId", "MLBAMID",
    ]]

for YEAR in range(startYear,EndYear+1):
    year_bwar = fetch_bwar(YEAR)
    sv_df = fetch_statcast_ev(YEAR)
    xw_df = fetch_expected_stats(YEAR)
    battracking_df = fetch_bat_tracking(YEAR)

    savant_statcast_df = sv_df.merge(xw_df, on="player_id", how="outer")
    savant_statcast_df["player_id"] = pd.to_numeric(savant_statcast_df["player_id"], errors="coerce")
    final = fetch_fangraphs_batting(YEAR)
    final.columns = final.columns.str.strip().str.replace('\ufeff', '')
    final["Year"] = YEAR
    final["MLBAMID"] = pd.to_numeric(final["MLBAMID"], errors="coerce")
    fielding = pd.read_csv(f"data/fielding_{YEAR}.csv")
    fielding.columns = fielding.columns.str.strip().str.replace('\ufeff', '')
    fielding["Inn"] = pd.to_numeric(fielding["Inn"], errors="coerce")
    primary_pos  = fielding.loc[fielding.groupby("PlayerId")["Inn"].idxmax(), ["PlayerId", "Pos"]]
    fielding_agg = fielding.groupby("PlayerId", as_index=False).sum(numeric_only=True).merge(primary_pos, on="PlayerId", how="left")
    fielding_agg.drop(columns=[c for c in fielding_agg.columns if c != "PlayerId" and c in final.columns], inplace=True)
    final = final.merge(fielding_agg, on="PlayerId", how="left")
    final.drop(columns=[
            "wOBA", "xwOBA", "xBA", "xSLG",
            "EV", "maxEV", "HardHit%", "Barrel%",
            "BatSpd", "SqUpSw%"
        ], errors="ignore", inplace=True)
    final = final.merge(savant_statcast_df, left_on="MLBAMID", right_on="player_id", how="left")

    final = final.merge(battracking_df, left_on="MLBAMID", right_on="player_id", how="left")
        

    final = final.merge(year_bwar[["MLBAMID", "bWAR_val"]], on="MLBAMID", how="left")
    final.rename(columns={"bWAR_val": "bWAR"}, inplace=True)
    final["bWAR"] = final["bWAR"].fillna(0)



    final["TB"]                = final["1B"] + final["2B"]*2 + final["3B"]*3 + final["HR"]*4
    final["XBH"]               = final["2B"] + final["3B"] + final["HR"]
    final["fWAR-bWAR Avg"]     = (final["fWAR"] + final["bWAR"]) / 2
    final["fWAR/650"]          = final["fWAR"] / final["PA"] * 650
    final["bWAR/650"]          = final["bWAR"] / final["PA"] * 650
    final["wOBA-xwOBA"]        = final["wOBA"] - final["xwOBA"]
    final["Whiff%"]          = 1 - final["Whiff%"]
    final["Z-Swing% - Chase%"] = final["Z-Swing%"] - final["Chase%"]
    final["DRS/1350"]          = (final["DRS"] / final["Inn"] * 1350).round(0)
    final["OAA/1350"]          = (final["OAA"] / final["Inn"] * 1350).round(0)
    final["FRV/1350"]          = (final["FRV"] / final["Inn"] * 1350).round(0)
    final["FRM/1350"]          = (final["FRM"] / final["Inn"] * 1350).round(1)

    for col in STAT_ALLOWLIST:
        if col not in final.columns:
            final[col] = None

    final = final[["Name", "PlayerId", "MLBAMID", "Pos", "Team"] + [col for col in STAT_ALLOWLIST]]

    if localUpload:
        final.to_csv(f"data/final/hitting_final_{YEAR}.csv")
        print(f"Locally uploaded {len(final)} players to data/final/hitting_final_{YEAR}.csv")
    
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    bucket = "sports-analytics-files"

    if upload:
        csv_buffer = StringIO()
        final.to_csv(csv_buffer, index=False)
        s3.put_object(
                    Bucket=bucket,
                    Key=f"processed/hitting_final_{YEAR}.csv",
                    Body=csv_buffer.getvalue().encode("utf-8"),
                )
        print(f"Uploaded {len(final)} players to s3://{bucket}/processed/hitting_final_{YEAR}.csv")

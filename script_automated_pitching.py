import pandas as pd
from p_utils import STAT_ALLOWLIST
from io import StringIO
import requests

from utils import BREF_COOKIE, FG_COOKIE, strip_html, _browser_headers, s3, bucket

localUpload = False

upload = True

startYear = 2004
EndYear = 2025

MULTIPLY_100_IF_DECIMAL = {
    "Whiff%", "Chase%", "K%", "BB%",
}


def fetch_bwar(year: int) -> pd.DataFrame:
    url = "https://www.baseball-reference.com/data/war_daily_pitch.txt"
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

def fetch_statcast_ev_pitching(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?csv=true&type=pitcher&year={year}&position=&team=&min=1"
        "&sort=barrels_per_pa&sortDir=asc"
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df[["player_id", "avg_hit_speed", "ev95percent", "brl_percent"]].rename(columns={
        "avg_hit_speed": "EV", "ev95percent": "HardHit%", "brl_percent": "Barrel%",
    })
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    for col in ["EV", "HardHit%", "Barrel%"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fetch_xera(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?csv=true&type=pitcher&year={year}&position=&team=&filterType=bip&min=1"
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df[["player_id", "xera"]].rename(columns={"xera": "xERA_sv"})
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["xERA_sv"] = pd.to_numeric(df["xERA_sv"], errors="coerce")
    return df

def fetch_fangraphs_pitching(year: int) -> pd.DataFrame:
    url = (
        "https://www.fangraphs.com/api/leaders/major-league/data"
        f"?age=&pos=all&stats=pit&lg=all&qual=0"
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
        "WAR":        "fWAR",
        "playerid":   "PlayerId",
        "xMLBAMID":   "MLBAMID",
        "PlayerName": "NameASCII",
        "pivFA":      "vFA",
        "scO-Swing%": "Chase%",
        "piContact%": "Whiff%",
    })
    return df[[
        "Name", "Team", "W", "L", "SV", "G", "GS", "IP",
        "K/9", "BB/9", "HR/9", "BABIP", "LOB%", "GB%", "HR/FB",
        "vFA", "ERA", "xERA", "FIP", "xFIP", "fWAR",
        "CG", "ShO", "TBF", "H", "ER", "HR", "BB", "IBB", "HBP", "SO", "QS",
        "AVG", "WHIP", "ERA-", "FIP-", "K%", "BB%", "SIERA", "K-BB%",
        "WPA", "Clutch",
        "EV", "Barrel%", "HardHit%", "K/BB",
        "Whiff%", "Chase%",
        "NameASCII", "PlayerId", "MLBAMID",
    ]]

for YEAR in range(startYear,EndYear+1):
    year_bwar = fetch_bwar(YEAR)

    if YEAR >= 2015:
        ev_df = fetch_statcast_ev_pitching(YEAR)
        xera_df = fetch_xera(YEAR)
        savant_df = ev_df.merge(xera_df, on="player_id", how="outer")
    else:
        savant_df = pd.DataFrame(columns=["player_id", "EV", "HardHit%", "Barrel%", "xERA_sv"])
   
    final = fetch_fangraphs_pitching(YEAR)
    final.columns = final.columns.str.strip().str.replace('\ufeff', '')
    final["Year"] = YEAR

    final.drop(columns=[
            "EV",
            "HardHit%",
            "Barrel%",
        ], errors="ignore", inplace=True)
    
    final = final.merge(savant_df, left_on="MLBAMID", right_on="player_id", how="left").drop(columns=["player_id"])
    
    final["MLBAMID"] = pd.to_numeric(final["MLBAMID"], errors="coerce")
    final = final.merge(year_bwar[["MLBAMID", "bWAR_val","Name"]], on="MLBAMID", how="left")
    final["Name"] = final["Name_y"].fillna(final["Name_x"])
    final.rename(columns={"bWAR_val": "bWAR"}, inplace=True)
    final["bWAR"] = final["bWAR"].fillna(0)
    final["fWAR-bWAR Avg"]     = (final["fWAR"] + final["bWAR"]) / 2
    final["fWAR/200"]          = final["fWAR"] / final["IP"] * 200
    final["bWAR/200"]          = final["bWAR"] / final["IP"] * 200
    final["Whiff%"]          = 1 - final["Whiff%"]
    final["xERA"]              = final["xERA_sv"].fillna(final["xERA"])
    final["ERA-xERA"]          = final["ERA"] - final["xERA"]
    final.drop(columns=["xERA_sv"], inplace=True)

    for col in MULTIPLY_100_IF_DECIMAL:
        if col in final.columns:
            numeric = pd.to_numeric(final[col], errors="coerce")
            if numeric.median() <= 1:
                final[col] = numeric * 100
        
    for col in STAT_ALLOWLIST:
        if col not in final.columns:
            final[col] = None

    final = final[["Name", "PlayerId", "MLBAMID", "Team"] + [col for col in STAT_ALLOWLIST]]

    if localUpload:
        final.to_csv(f"data/final/pitching_final_{YEAR}.csv")
        print(f"Locally uploaded {len(final)} players to data/final/pitching_final_{YEAR}.csv")

    if upload:
        csv_buffer = StringIO()
        final.to_csv(csv_buffer, index=False)
        s3.put_object(
                    Bucket=bucket,
                    Key=f"processed/pitching_final_{YEAR}.csv",
                    Body=csv_buffer.getvalue().encode("utf-8"),
                )
        print(f"Uploaded {len(final)} players to s3://{bucket}/processed/pitching_final_{YEAR}.csv")

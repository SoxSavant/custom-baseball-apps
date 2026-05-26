import pandas as pd
from pathlib import Path

from h_utils import STAT_ALLOWLIST

YEAR = 2026
LOCAL_BWAR_FILE = Path("war_daily_bat.txt")


sv_df = pd.read_csv(f"data/sweetspot_{YEAR}.csv")
sv_df = sv_df[["player_id", "anglesweetspotpercent", "avg_hit_speed", "max_hit_speed", "ev95percent", "brl_percent"]].rename(columns={
    "avg_hit_speed": "EV", "max_hit_speed": "maxEV", "ev95percent": "HardHit%", "brl_percent": "Barrel%",
})

xw_df = pd.read_csv(f"data/xwoba_{YEAR}.csv")
xw_df = xw_df[["player_id", "est_ba", "est_slg", "est_woba", "woba"]].rename(columns={
    "est_ba": "xBA", "est_slg": "xSLG", "est_woba": "xwOBA", "woba": "wOBA",
})


savant_statcast_df = sv_df.merge(xw_df, on="player_id", how="outer")
savant_statcast_df["player_id"] = pd.to_numeric(savant_statcast_df["player_id"], errors="coerce")

bt_df = pd.read_csv(f"data/bat-tracking_{YEAR}.csv")

savant_battracking_df = bt_df[["id", "avg_bat_speed", "squared_up_per_swing"]].rename(columns={
    "avg_bat_speed": "BatSpd", "squared_up_per_swing": "SqUpSw%",
})


bwar_df = pd.read_csv(LOCAL_BWAR_FILE)
bwar_df["MLBAMID"]  = pd.to_numeric(bwar_df.get("mlb_ID"),  errors="coerce")
bwar_df["year_ID"]  = pd.to_numeric(bwar_df.get("year_ID"), errors="coerce")
bwar_df["bWAR_val"] = pd.to_numeric(bwar_df.get("WAR"),     errors="coerce")
bwar_df = bwar_df.dropna(subset=["MLBAMID", "year_ID", "bWAR_val"])
year_bwar = bwar_df.groupby(["MLBAMID", "year_ID"], as_index=False)["bWAR_val"].sum()
year_bwar = year_bwar[year_bwar["year_ID"] == YEAR]

final = pd.read_csv(f"data/batting_{YEAR}.csv")
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

final = final.merge(savant_battracking_df, left_on="MLBAMID", right_on="id", how="left")
final = final.merge(year_bwar[["MLBAMID", "bWAR_val"]], on="MLBAMID", how="left")
final.rename(columns={"bWAR_val": "bWAR"}, inplace=True)
final["bWAR"] = final["bWAR"].fillna(0)

final["TB"]                = final["1B"] + final["2B"]*2 + final["3B"]*3 + final["HR"]*4
final["XBH"]               = final["2B"] + final["3B"] + final["HR"]
final["fWAR-bWAR Avg"]     = (final["WAR"] + final["bWAR"]) / 2
final["fWAR/650"]          = final["WAR"] / final["PA"] * 650
final["bWAR/650"]          = final["bWAR"] / final["PA"] * 650
final["wOBA-xwOBA"]        = final["wOBA"] - final["xwOBA"]
final["Contact%"]          = 1 - final["Contact%"]
final["Z-Swing% - Chase%"] = final["Z-Swing%"] - final["O-Swing%"]
final["DRS/1350"]          = (final["DRS"] / final["Inn"] * 1350).round(0)
final["OAA/1350"]          = (final["OAA"] / final["Inn"] * 1350).round(0)
final["FRV/1350"]          = (final["FRV"] / final["Inn"] * 1350).round(0)
final["FRM/1350"]          = (final["FRM"] / final["Inn"] * 1350).round(1)

final.rename(columns={
    "WAR": "fWAR", "O-Swing%": "Chase%", "Contact%": "Whiff%",
    "SqUpSw%": "Squared-Up%", "anglesweetspotpercent": "Sweet-Spot%",
}, inplace=True)

for col in STAT_ALLOWLIST:
    if col not in final.columns:
        final[col] = None

final = final[["Name", "PlayerId", "MLBAMID", "Pos", "Team"] + [col for col in STAT_ALLOWLIST]]

Path("data/final").mkdir(parents=True, exist_ok=True)
final.to_csv(f"data/final/hitting_final_{YEAR}.csv", index=False)


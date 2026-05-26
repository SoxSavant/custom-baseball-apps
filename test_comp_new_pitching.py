import pandas as pd
from pathlib import Path

from p_utils import STAT_ALLOWLIST

YEAR = 2026
LOCAL_BWAR_FILE = Path("war_daily_pitch.txt")

ev_df = pd.read_csv(f"data/ev_{YEAR}.csv")
ev_df = ev_df[["player_id", "avg_hit_speed", "ev95percent", "brl_percent"]].rename(columns={
    "avg_hit_speed": "EV", "ev95percent": "HardHit%", "brl_percent": "Barrel%",
})

xera_df = pd.read_csv(f"data/xera_{YEAR}.csv")
xera_df = xera_df[["player_id", "xera"]].rename(columns={"xera": "xERA_sv"})

savant_df = ev_df.merge(xera_df, on="player_id", how="outer")

bwar_df = pd.read_csv(LOCAL_BWAR_FILE)
bwar_df["MLBAMID"]  = pd.to_numeric(bwar_df.get("mlb_ID"),  errors="coerce")
bwar_df["year_ID"]  = pd.to_numeric(bwar_df.get("year_ID"), errors="coerce")
bwar_df["bWAR_val"] = pd.to_numeric(bwar_df.get("WAR"),     errors="coerce")
bwar_df = bwar_df.dropna(subset=["MLBAMID", "year_ID", "bWAR_val"])
year_bwar = bwar_df.groupby(["MLBAMID", "year_ID"], as_index=False)["bWAR_val"].sum()
year_bwar = year_bwar[year_bwar["year_ID"] == YEAR]

final = pd.read_csv(f"data/pitching_{YEAR}.csv")
final.columns = final.columns.str.strip().str.replace('\ufeff', '')
final["Year"] = YEAR

final.drop(columns=[
    "EV",
    "HardHit%",
    "Barrel%",
], errors="ignore", inplace=True)

final = final.merge(savant_df, left_on="MLBAMID", right_on="player_id", how="left").drop(columns=["player_id"])
final["MLBAMID"] = pd.to_numeric(final["MLBAMID"], errors="coerce")
final = final.merge(year_bwar[["MLBAMID", "bWAR_val"]], on="MLBAMID", how="left")
final.rename(columns={"bWAR_val": "bWAR"}, inplace=True)
final["bWAR"] = final["bWAR"].fillna(0)

final["fWAR-bWAR Avg"]     = (final["WAR"] + final["bWAR"]) / 2
final["fWAR/200"]          = final["WAR"] / final["IP"] * 200
final["bWAR/200"]          = final["bWAR"] / final["IP"] * 200
final["Contact%"]          = 1 - final["Contact%"]
final["xERA"]              = final["xERA_sv"].fillna(final["xERA"])
final["ERA-xERA"]          = final["ERA"] - final["xERA"]
final.drop(columns=["xERA_sv"], inplace=True)


final.rename(columns={
    "WAR": "fWAR", "O-Swing%": "Chase%", "Contact%": "Whiff%", "vFA (pi)": "vFA",
}, inplace=True)

for col in STAT_ALLOWLIST:
    if col not in final.columns:
        final[col] = None

final = final[["Name", "PlayerId", "MLBAMID", "Team"] + [col for col in STAT_ALLOWLIST]]

Path("data/final").mkdir(parents=True, exist_ok=True)
final.to_csv(f"data/final/pitching_final_{YEAR}.csv", index=False)

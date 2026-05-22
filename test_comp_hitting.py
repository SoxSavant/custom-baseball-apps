import pandas as pd
from pathlib import Path

from h_utils import STAT_ALLOWLIST

keep_cols = ["Name", "PlayerId","MLBAMID","Pos","Team"]

# --- Configuration ---
LOCAL_BWAR_FILE = Path("war_daily_bat.txt") 

def load_bwar_master() -> pd.DataFrame:
    """Loads the entire BRef history and aggregates by ID and Year."""
    if not LOCAL_BWAR_FILE.exists():
        print("Warning: bWAR file not found.")
        return pd.DataFrame()
    
    # Read raw data
    df = pd.read_csv(LOCAL_BWAR_FILE)
    
    # Standardize columns to match your merge keys
    df["MLBAMID"] = pd.to_numeric(df.get("mlb_ID"), errors="coerce")
    df["year_ID"] = pd.to_numeric(df.get("year_ID"), errors="coerce")
    df["bWAR_val"] = pd.to_numeric(df.get("WAR"), errors="coerce")
    
    # Clean and Aggregate: This handles players with multiple rows per season (trades)
    df = df.dropna(subset=["MLBAMID", "year_ID", "bWAR_val"])
    df_agg = df.groupby(["MLBAMID", "year_ID"], as_index=False)["bWAR_val"].sum()
    
    return df_agg

bwar_master = load_bwar_master()

for year in range(1901, 2027):

    hitting_dfs = [
        pd.read_csv(f"data/batting_{year}.csv"),
        pd.read_csv(f"data/standard_{year}.csv"),
        pd.read_csv(f"data/advanced_{year}.csv"),
    ]

    
    
    if year >=2023: #bat speed data started in 2023
        hitting_dfs.append(pd.read_csv(f"data/batspeed_{year}.csv"))
    if year >=2015: #statcast data started in 2015
        hitting_dfs.append(pd.read_csv(f"data/statcast_{year}.csv"))
    if year >=2007: #discipline data started in 2015
        hitting_dfs.append(pd.read_csv(f"data/discipline_{year}.csv"))
    if year >= 1974: # win prob started in 1974
        hitting_dfs.append(pd.read_csv(f"data/winprob_{year}.csv"))

    base_cols = {"Name", "Team", "MLBAMID", "NameASCII"}

    for df in hitting_dfs:
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('\ufeff', '')

    # Build merged result iteratively, dropping duplicate cols before each merge
    hitting_merged = hitting_dfs[0]
    for df in hitting_dfs[1:]:
        df = df.drop(columns=base_cols, errors="ignore")

        # Drop cols already in the merged frame (except the join key)
        duplicate_cols = [
            c for c in df.columns
            if c != "PlayerId" and c != "SqUpSw%" and c in hitting_merged.columns
        ]
        df = df.drop(columns=duplicate_cols)

        hitting_merged = hitting_merged.merge(df, on="PlayerId", how="left")
    if year >=2015: #for special baseball savant csv which has player_id, not PlayerId
        ss_df = pd.read_csv(f"data/sweetspot_{year}.csv")
        hitting_merged = hitting_merged.merge(ss_df, left_on = "MLBAMID", right_on = "player_id", how = "left").drop(columns=["player_id"], errors="ignore")

    # Fielding: aggregate across positions, then drop cols already in hitting
    fielding = pd.read_csv(f"data/fielding_{year}.csv")
    fielding.columns = fielding.columns.str.strip()
    fielding.columns = fielding.columns.str.replace('\ufeff', '')

    if year >=1956: # Inn starts in 1956
        fielding["Inn"] = pd.to_numeric(fielding["Inn"], errors="coerce")


        idx = fielding.groupby("PlayerId")["Inn"].idxmax()
        primary_pos = fielding.loc[idx, ["PlayerId", "Pos"]]
    else:
        idx = fielding.groupby("PlayerId")["G"].idxmax()
        primary_pos = fielding.loc[idx, ["PlayerId", "Pos"]]

    numeric_agg = fielding.groupby("PlayerId", as_index=False).sum(numeric_only=True)


    fielding_agg = numeric_agg.merge(primary_pos, on="PlayerId", how="left")

    duplicate_fielding_cols = [
        c for c in fielding_agg.columns
        if c != "PlayerId" and c in hitting_merged.columns
    ]
    fielding_agg = fielding_agg.drop(columns=duplicate_fielding_cols)

    final = hitting_merged.merge(fielding_agg, on="PlayerId", how="left")
    final["Year"] = year

    if not bwar_master.empty:
        # Filter for the specific year in the loop
        year_bwar = bwar_master[bwar_master["year_ID"] == year].copy()
        
        # Merge onto your main dataframe using MLBAMID
        final = final.merge(
            year_bwar[["MLBAMID", "bWAR_val"]], 
            on="MLBAMID", 
            how="left"
        )
        
        final.rename(columns={"bWAR_val": "bWAR"}, inplace=True)
        final["bWAR"] = final["bWAR"].fillna(0)

    
    final["TB"] = final["1B"] + final["2B"]*2 + final["3B"]*3 + final["HR"]*4
    final["XBH"] = final["2B"]+ final["3B"] + final["HR"]
    final["fWAR-bWAR Avg"] = (final["WAR"] + final["bWAR"]) / 2
    if "WAR" and "PA" in final.columns:
        final["fWAR/650"] = final["WAR"] / final["PA"] * 650
    if "bWAR" and "PA" in final.columns:
        final["bWAR/650"] = final["bWAR"] / final["PA"] * 650
    if "wOBA" in final.columns and "xwOBA" in final.columns:
        final["wOBA-xwOBA"] = final["wOBA"] - final["xwOBA"]
    if "Contact%" in final.columns:
        final["Contact%"] = 1 - final["Contact%"]

    if "DRS" in final.columns and "Inn" in final.columns:
        final["DRS/1350"] = (final["DRS"] / final["Inn"] * 1350).round(0)
    if "OAA" in final.columns and "Inn" in final.columns:
        final["OAA/1350"] = (final["OAA"] / final["Inn"] * 1350).round(0)
    if "FRV" in final.columns and "Inn" in final.columns:
        final["FRV/1350"] = (final["FRV"] / final["Inn"] * 1350).round(0)
    if "FRM" in final.columns and "Inn" in final.columns:
        final["FRM/1350"] = (final["FRM"] / final["Inn"] * 1350).round(1)
    if "O-Swing%" in final.columns and "Z-Swing%" in final.columns:
        final["Z-Swing% - Chase%"] = final["Z-Swing%"] - final["O-Swing%"]
    
    
    cols_to_drop = [c for c in ["fWAR", "Chase%", "Whiff%", "Squared-Up%"] 
                if c in final.columns]
    final = final.drop(columns=cols_to_drop)
    
    rename_map = {}
    if "WAR" in final.columns:
        rename_map["WAR"] = "fWAR"
    if "O-Swing%" in final.columns:
        rename_map["O-Swing%"] = "Chase%"
    if "Contact%" in final.columns:
        rename_map["Contact%"] = "Whiff%"
    if "SqUpSw%" in final.columns:
        rename_map["SqUpSw%"] = "Squared-Up%"
    if "anglesweetspotpercent" in final.columns:
        rename_map["anglesweetspotpercent"] = "Sweet-Spot%"

    final.rename(columns=rename_map, inplace=True)

    for col in STAT_ALLOWLIST:
        if col not in final.columns:
            final[col] = None

    
    final = final[keep_cols + [col for col in STAT_ALLOWLIST]] # drop columns after renaming them to ones in allow list
    
    final.to_csv(f"data/final/hitting_final_{year}.csv", index=False)
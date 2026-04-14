import pandas as pd
from pathlib import Path

# --- Configuration ---
LOCAL_BWAR_FILE = Path("warhitters2025.txt") 

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

for year in range(2000, 2027):

    hitting_dfs = [
        pd.read_csv(f"data/batting_{year}.csv"),
        pd.read_csv(f"data/standard_{year}.csv"),
        pd.read_csv(f"data/advanced_{year}.csv"),
        pd.read_csv(f"data/winprob_{year}.csv"),
    ]

    
    if year >=2023: #bat speed data started in 2023
        hitting_dfs.append(pd.read_csv(f"data/batspeed_{year}.csv"))
    if year>=2015: #statcast data started in 2015
        hitting_dfs.append(pd.read_csv(f"data/statcast_{year}.csv"))
    if year>=2007: #discipline data started in 2015
        hitting_dfs.append(pd.read_csv(f"data/discipline_{year}.csv"))

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
            if c != "PlayerId" and c in hitting_merged.columns
        ]
        df = df.drop(columns=duplicate_cols)

        hitting_merged = hitting_merged.merge(df, on="PlayerId", how="left")

    # Fielding: aggregate across positions, then drop cols already in hitting
    fielding = pd.read_csv(f"data/fielding_{year}.csv")
    fielding.columns = fielding.columns.str.strip()
    fielding.columns = fielding.columns.str.replace('\ufeff', '')

   
    fielding["Inn"] = pd.to_numeric(fielding["Inn"], errors="coerce")


    idx = fielding.groupby("PlayerId")["Inn"].idxmax()
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
        
        # Rename to your preferred column name and fill missing with 0
        final.rename(columns={"bWAR_val": "bWAR"}, inplace=True)
        final["bWAR"] = final["bWAR"].fillna(0)

    if year>=2007:
        final["Contact%"] = 1 - final["Contact%"]
    final["TB"] = final["1B"] + final["2B"]*2 + final["3B"]*3 + final["HR"]*4
    final["XBH"] = final["2B"]+ final["3B"] + final["HR"]
    if year >=2025:
        final["wOBA-xwOBA"] = final["wOBA"] - final["xwOBA"]
    if year < 2023:
        final["BatSpd"] = None
    if year < 2015:
        final["EV"] = None
        final["Barrel%"] = None
        final["HardHit%"] = None
        final["xBA"] = None
        final["xSLG"] = None
        final["xwOBA"] = None
        final["maxEV"] = None
    if year < 2016:
        final["FRV"] = None
        final["OAA"] = None
    if year < 2007:
        final["O-Swing%"] = None
        final["Z-Swing%"] = None
        final["Swing%"] = None
        final["O-Contact%"] = None
        final["Z-Contact%"] = None
        final["Contact%"] = None
        final["Zone%"] = None
    if year >=2002: # TZ records ended in 2001, DRS started in 2002
        final["TZ"] = None
    else:
        final["DRS"] = None

    final.rename(columns={"Contact%":"Whiff%", "O-Swing%":"Chase%", "WAR":"fWAR"}, inplace=True)
    final.to_csv(f"data/final/hitting_final_{year}.csv", index=False)
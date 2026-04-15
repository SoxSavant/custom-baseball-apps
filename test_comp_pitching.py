import pandas as pd
from pathlib import Path


LOCAL_BWAR_FILE = Path("war_daily_pitch.txt") 

def load_bwar_master() -> pd.DataFrame:
    """Loads the entire BRef pitching history and aggregates by ID and Year."""
    if not LOCAL_BWAR_FILE.exists():
        print(f"Warning: {LOCAL_BWAR_FILE} not found.")
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

    pitching_dfs = [
        pd.read_csv(f"data/pitching_{year}.csv"),
        pd.read_csv(f"data/pitching_advanced_{year}.csv"),
        pd.read_csv(f"data/pitching_standard_{year}.csv"),
        pd.read_csv(f"data/pitching_winprob_{year}.csv"),
    ]

    if year>=2015: #statcast data started in 2015
        pitching_dfs.append(pd.read_csv(f"data/pitching_statcast_{year}.csv"))

    if year >=2007: #plate discipline data started in 2007
        pitching_dfs.append(pd.read_csv(f"data/discipline_pitching_{year}.csv"))

    base_cols = {"Name", "Team", "MLBAMID", "NameASCII"}

    for df in pitching_dfs:
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('\ufeff', '')

    # Build merged result iteratively, dropping duplicate cols before each merge
    pitching_merged = pitching_dfs[0]
    for df in pitching_dfs[1:]:
        df = df.drop(columns=base_cols, errors="ignore")

        # Drop cols already in the merged frame (except the join key)
        duplicate_cols = [
            c for c in df.columns
            if c != "PlayerId" and c in pitching_merged.columns
        ]
        df = df.drop(columns=duplicate_cols)

        pitching_merged = pitching_merged.merge(df, on="PlayerId", how="left")


    final = pitching_merged
    final["Year"] = year

    if not bwar_master.empty:
        # Ensure MLBAMID is numeric for matching
        final["MLBAMID"] = pd.to_numeric(final["MLBAMID"], errors="coerce")
        
        # Filter master bWAR for current year
        year_bwar = bwar_master[bwar_master["year_ID"] == year].copy()
        
        # Merge
        final = final.merge(
            year_bwar[["MLBAMID", "bWAR_val"]], 
            on="MLBAMID", 
            how="left"
        )
        
        # Finalize bWAR column
        final.rename(columns={"bWAR_val": "bWAR"}, inplace=True)
        final["bWAR"] = final["bWAR"].fillna(0)

    if year >=2007:
        final["Contact%"] = 1 - final["Contact%"]
    
    if year < 2015:
        final["EV"] = None
        final["Barrel%"] = None
        final["HardHit%"] = None
        final["xERA"] = None
    if year < 2007:
        final["O-Swing%"] = None
        final["Z-Swing%"] = None
        final["Swing%"] = None
        final["O-Contact%"] = None
        final["Z-Contact%"] = None
        final["Contact%"] = None
        final["Zone%"] = None
    
    final.rename(columns={"Contact%":"Whiff%", "O-Swing%":"Chase%","WAR":"fWAR","vFA (pi)": "vFA"}, inplace=True)
    final.to_csv(f"data/final/pitching_final_{year}.csv", index=False)
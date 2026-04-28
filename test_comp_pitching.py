import pandas as pd
from pathlib import Path

from p_utils import STAT_ALLOWLIST

keep_cols = ["Name", "PlayerId","MLBAMID","Team"]

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

for year in range(2026, 2027):

    pitching_dfs = [
        pd.read_csv(f"data/pitching_{year}.csv"),
        pd.read_csv(f"data/pitching_advanced_{year}.csv"),
        pd.read_csv(f"data/pitching_standard_{year}.csv"),
    ]

    if year>=2015: #statcast data started in 2015
        pitching_dfs.append(pd.read_csv(f"data/pitching_statcast_{year}.csv"))

    if year >=2007: #plate discipline data started in 2007
        pitching_dfs.append(pd.read_csv(f"data/discipline_pitching_{year}.csv"))
    if year >= 1974: # win prob started in 1974
        pitching_dfs.append(pd.read_csv(f"data/pitching_winprob_{year}.csv"))

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

    if "Contact%" in final.columns:
        final["Contact%"] = 1 - final["Contact%"]
    for col in STAT_ALLOWLIST:
        if col not in final.columns:
            final[col] = None
    
    final.rename(columns={"Contact%":"Whiff%", "O-Swing%":"Chase%","WAR":"fWAR","vFA (pi)": "vFA"}, inplace=True)

    final = final[keep_cols + [col for col in STAT_ALLOWLIST]] # only drop columns AFTER renaming them to ones in allow list
    
    final.to_csv(f"data/final/pitching_final_{year}.csv", index=False)
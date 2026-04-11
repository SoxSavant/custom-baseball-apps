import pandas as pd
import os

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
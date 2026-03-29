import pandas as pd
from functools import reduce

for year in range(2015, 2026):

    pitching_dfs = [
        pd.read_csv(f"data/pitching_{year}.csv"),
        pd.read_csv(f"data/pitching_advanced_{year}.csv"),
        pd.read_csv(f"data/pitching_standard_{year}.csv"),
        pd.read_csv(f"data/pitching_statcast_{year}.csv"),
        pd.read_csv(f"data/pitching_winprob_{year}.csv"),
        pd.read_csv(f"data/discipline_pitching_{year}.csv"),

    ]

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
    final["Contact%"] = 1 - final["Contact%"]
    final.rename(columns={"Contact%":"Whiff%", "O-Swing%":"Chase%","WAR":"fWAR"}, inplace=True)
    final.to_csv(f"data/final/pitching_final_{year}.csv", index=False)
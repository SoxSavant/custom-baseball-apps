import pandas as pd
from functools import reduce

for year in range(2015, 2026):

    hitting_dfs = [
        pd.read_csv(f"data/batting_{year}.csv"),
        pd.read_csv(f"data/standard_{year}.csv"),
        pd.read_csv(f"data/advanced_{year}.csv"),
        pd.read_csv(f"data/statcast_{year}.csv"),
        pd.read_csv(f"data/winprob_{year}.csv"),
        pd.read_csv(f"data/discipline_{year}.csv"),
    ]

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
    final.to_csv(f"data/final/hitting_final_{year}.csv", index=False)
"""
Finds every year where at least one player was top-5 in the league in BOTH
Off and Def, among players with at least MIN_PA plate appearances.

Uses h_utils.load_final_year, so run this from the same directory as your
h_utils.py (or make sure it's importable).
"""

import pandas as pd
from h_utils import load_final_year, start_year

MIN_PA = 350
TOP_N = 3
START_YEAR = start_year
END_YEAR = 2026
STATS = ["Off", "Def"]


def top_n_overlap_for_year(year: int, min_pa: int = MIN_PA, top_n: int = TOP_N):
    df = load_final_year(year)
    if df is None or df.empty:
        return None

    for col in STATS + ["PA"]:
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["PA"] >= min_pa].dropna(subset=STATS)
    if df.empty:
        return None

    # rank with method="min" so ties all count toward the top N
    off_rank = df["Off"].rank(ascending=False, method="min")
    def_rank = df["Def"].rank(ascending=False, method="min")

    top_off = set(df.loc[off_rank <= top_n, "PlayerId"])
    top_def = set(df.loc[def_rank <= top_n, "PlayerId"])

    overlap_ids = top_off & top_def
    if not overlap_ids:
        return None

    hits = []
    for pid in overlap_ids:
        mask = df["PlayerId"] == pid
        row = df[mask].iloc[0]
        hits.append({
            "Name": row["Name"],
            "Off": row["Off"],
            "Off_rank": int(off_rank[mask].iloc[0]),
            "Def": row["Def"],
            "Def_rank": int(def_rank[mask].iloc[0]),
        })
    return hits


def main():
    results = {}
    for year in range(START_YEAR, END_YEAR + 1):
        print(year, flush=True)
        hits = top_n_overlap_for_year(year)
        if hits:
            results[year] = hits
            for h in hits:
                print(
                    f"  -> {h['Name']}: Off {h['Off']:.1f} (#{h['Off_rank']}), "
                    f"Def {h['Def']:.1f} (#{h['Def_rank']})",
                    flush=True,
                )

    if not results:
        print(f"\nDone. No player was top-{TOP_N} in both Off and Def (min {MIN_PA} PA) in any year checked.")
        return

    print(f"\nDone. Total years with overlap: {len(results)}")

    print("\n--- Summary (chronological) ---\n")
    for year in sorted(results):
        for h in results[year]:
            print(
                f"{year}: {h['Name']}: Off {h['Off']:.1f} (#{h['Off_rank']}), "
                f"Def {h['Def']:.1f} (#{h['Def_rank']})\n"
            )


if __name__ == "__main__":
    main()
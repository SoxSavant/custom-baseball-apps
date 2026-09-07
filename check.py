"""
For every year 1901-2026, uses load_combined_year() from warleaders.py
(hitting fWAR + pitching fWAR already summed per player there) to find the
gap between the #1 and #2 player that year, then prints the 10 biggest gaps
across all years.

Run from the same directory as warleaders.py / h_utils.py / p_utils.py,
with AWS env vars set.
"""

from warleaders import load_combined_year

START_YEAR = 1947
END_YEAR = 2026


def gap_for_year(year: int):
    df = load_combined_year(year)
    if df is None or df.empty or "fWAR" not in df.columns:
        return None

    df = df.dropna(subset=["fWAR"])
    if len(df) < 2:
        return None

    top2 = df.sort_values("fWAR", ascending=False).head(2)
    first, second = top2.iloc[0], top2.iloc[1]
    return {
        "year": year,
        "p1_name": first["Name"], "p1_war": first["fWAR"],
        "p2_name": second["Name"], "p2_war": second["fWAR"],
        "gap": first["fWAR"].round(1) - second["fWAR"].round(1),
    }


def main():
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(year, flush=True)
        r = gap_for_year(year)
        if r:
            rows.append(r)

    if not rows:
        print("No data found.")
        return

    rows.sort(key=lambda r: r["gap"], reverse=True)

    print("\nTop 10 biggest #1 vs #2 fWAR gaps, 1901-2026")
    for r in rows[:10]:
        print(
            f"{r['year']}: {r['p1_name']} ({r['p1_war']:.1f}) vs "
            f"{r['p2_name']} ({r['p2_war']:.1f}) - gap {r['gap']:.1f}"
        )


if __name__ == "__main__":
    main()
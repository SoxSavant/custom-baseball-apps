"""
precompute_yoy.py
-----------------
Precomputes year-over-year deltas for every consecutive season pair in MLB history.
Saves results to a CSV you can query instantly in your risers app or elsewhere.

Usage:
    python precompute_yoy.py

Output:
    yoy_deltas.csv  — one row per player per consecutive year pair
                      columns: Name, IDfg, start_year, end_year,
                               TeamDisplay_start, TeamDisplay_end,
                               PA_start, PA_end,
                               + delta columns for every stat (e.g. WAR, HR, wRC+, etc.)

Runtime: ~20-40 minutes depending on your connection (hits FanGraphs for each year).
Re-run at the start of each season to add the newest year.
"""

import pandas as pd
import numpy as np
import unicodedata
import pybaseball
import time
import re
from pathlib import Path

# ----------------------------
#  CONFIG
# ----------------------------
START_YEAR = 1955   # FanGraphs advanced stats get sparse before mid-50s
END_YEAR   = 2025
MIN_PA     = 600      # No filter here — filter at query time
OUTPUT_FILE = Path("yoy_deltas.csv")

# Stats to compute deltas for — everything meaningful in FanGraphs batting
STAT_COLS = [
    "WAR", "Off", "Def", "BsR",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG",
    "OPS", "SLG", "OBP", "AVG", "ISO", "BABIP",
    "PA", "AB", "G", "R", "RBI", "HR", "SB", "BB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Contact%",
    "Barrel%", "HardHit%", "EV",
    "GB%", "FB%", "LD%", "Pull%",
    "WPA", "Clutch",
]

# ----------------------------
#  HELPERS
# ----------------------------

def normalize_team_code(team: str, year: int):
    if not team:
        return None
    team = team.upper().strip()
    if team in {"", "-", "--", "---", "- - -", "TOT"}:
        return None
    if year < 2025:
        if team in {"ATH", "OAK"}:
            return "OAK"
    else:
        if team in {"ATH", "OAK"}:
            return "ATH"
    return team


def load_year(year: int) -> pd.DataFrame:
    """Load one season of batting stats, collapse multi-team players to one row."""
    print(f"  Loading {year}...", end=" ", flush=True)
    try:
        df = pybaseball.batting_stats(year, year, qual=MIN_PA, split_seasons=False)
    except Exception as e:
        print(f"FAILED ({e})")
        return pd.DataFrame()

    if df is None or df.empty:
        print("empty")
        return pd.DataFrame()

    df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")

    # Resolve TOT rows
    tot_ids = set(df.loc[df["Team"] == "TOT", "IDfg"])
    has_individual = set(df.loc[df["Team"] != "TOT", "IDfg"])
    tot_only_ids = tot_ids - has_individual
    non_tot = df[df["Team"] != "TOT"]
    tot_fallback = df[(df["Team"] == "TOT") & (df["IDfg"].isin(tot_only_ids))]
    df = pd.concat([non_tot, tot_fallback], ignore_index=True)
    df = df[df["Team"].notna()]

    # Collapse to one row per player
    collapsed = []
    for fg_id, grp in df.groupby("IDfg"):
        raw_teams = grp["Team"].dropna().astype(str).str.strip().str.upper().tolist()
        teams = [normalize_team_code(t, year) for t in raw_teams if t not in {"TOT", "---", "--", "-", ""}]
        teams = sorted(set(t for t in teams if t))

        tot_row = grp[grp["Team"] == "TOT"]
        base = tot_row.iloc[0].to_dict() if not tot_row.empty else grp.iloc[0].to_dict()

        base["TeamDisplay"] = teams[0] if len(teams) == 1 else ("2+ Teams" if teams else "---")
        base["Season"] = year
        collapsed.append(base)

    result = pd.DataFrame(collapsed)
    print(f"{len(result)} players")
    return result


# ----------------------------
#  MAIN
# ----------------------------

def main():
    print(f"Precomputing YoY deltas from {START_YEAR} to {END_YEAR}")
    print("=" * 50)

    # Load all years
    seasons = {}
    for year in range(START_YEAR, END_YEAR + 1):
        df = load_year(year)
        if not df.empty:
            df["IDfg"] = pd.to_numeric(df["IDfg"], errors="coerce")
            seasons[year] = df.set_index("IDfg")
        time.sleep(0.5)  # be polite to FanGraphs

    print(f"\nLoaded {len(seasons)} seasons. Computing deltas...")

    all_rows = []
    year_list = sorted(seasons.keys())

    for i in range(len(year_list) - 1):
        y_start = year_list[i]
        y_end   = year_list[i + 1]

        # Only consecutive years
        if y_end - y_start != 1:
            continue

        df_s = seasons[y_start]
        df_e = seasons[y_end]

        common_ids = set(df_s.index.dropna()) & set(df_e.index.dropna())
        if not common_ids:
            continue

        print(f"  {y_start}→{y_end}: {len(common_ids)} players in both years")

        for fg_id in common_ids:
            row_s = df_s.loc[fg_id]
            row_e = df_e.loc[fg_id]

            # Handle duplicate index (shouldn't happen but just in case)
            if isinstance(row_s, pd.DataFrame):
                row_s = row_s.iloc[0]
            if isinstance(row_e, pd.DataFrame):
                row_e = row_e.iloc[0]

            record = {
                "Name":             row_e.get("Name", row_s.get("Name", "")),
                "IDfg":             fg_id,
                "start_year":       y_start,
                "end_year":         y_end,
                "TeamDisplay_start": row_s.get("TeamDisplay", ""),
                "TeamDisplay_end":   row_e.get("TeamDisplay", ""),
                "PA_start":         pd.to_numeric(row_s.get("PA", np.nan), errors="coerce"),
                "PA_end":           pd.to_numeric(row_e.get("PA", np.nan), errors="coerce"),
            }

            for col in STAT_COLS:
                if col == "PA":
                    continue  # already stored above
                try:
                    s_val = pd.to_numeric(row_s.get(col, np.nan), errors="coerce")
                    e_val = pd.to_numeric(row_e.get(col, np.nan), errors="coerce")
                    record[f"{col}_start"] = s_val
                    record[f"{col}_end"]   = e_val
                    record[f"{col}_delta"] = e_val - s_val
                except Exception:
                    record[f"{col}_start"] = np.nan
                    record[f"{col}_end"]   = np.nan
                    record[f"{col}_delta"] = np.nan

            all_rows.append(record)

    print(f"\nTotal rows: {len(all_rows)}")
    result = pd.DataFrame(all_rows)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    print("\nDone! Example queries:")
    print("  Biggest WAR drops, min 600 PA each year:")
    print("    df = pd.read_csv('yoy_deltas.csv')")
    print("    filtered = df[(df.PA_start >= 600) & (df.PA_end >= 600)]")
    print("    filtered.nsmallest(10, 'WAR_delta')[['Name','start_year','end_year','WAR_start','WAR_end','WAR_delta']]")


if __name__ == "__main__":
    main()
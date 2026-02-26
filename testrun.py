import pandas as pd

df = pd.read_csv("yoy_deltas.csv")

# Biggest single-year WAR drops
result = df.nsmallest(20, "WAR_delta")[["Name", "start_year", "end_year", "WAR_start", "WAR_end", "WAR_delta"]]
print(result.to_string())
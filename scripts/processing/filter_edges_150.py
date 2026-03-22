import pandas as pd

df = pd.read_csv("data_main/edges.csv")
print(f"Total edges: {len(df):,}")
df150 = df[df["distance_km"] < 150]
print(f"After 150 km filter: {len(df150):,}")
df150.to_csv("data_main/edges_150.csv", index=False)
print("Saved data_main/edges_150.csv")

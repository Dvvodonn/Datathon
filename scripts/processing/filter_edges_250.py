import pandas as pd

df = pd.read_csv("data_main/edges.csv")
print(f"Total edges: {len(df):,}")

df250 = df[df["distance_km"] < 250]
print(f"After 250 km filter: {len(df250):,}")
df250.to_csv("data_main/edges_250.csv", index=False)
print("Saved data_main/edges_250.csv")

"""
enrich_nodes_capacity.py
========================
Adds n_chargers and mean_power_kw to data_main/nodes.csv for every row
where is_existing_charger == 1.

Source: data/raw/chargers/sites.csv  (DGT DATEX2 XML parsed to CSV)
  n_chargers    = number of non-null refillPoint[N]/connector[1]/connectorType[1] columns
  mean_power_kw = mean of refillPoint[N]/connector[1]/maxPowerAtSocket[1] / 1000

Match: exact lat/lon (all 2862 existing charger nodes come from the same DGT source;
verified match distance = 0 m for all rows).

Run from project root:
    python scripts/processing/enrich_nodes_capacity.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SITES_CSV = "data/raw/chargers/sites.csv"
NODES_CSV = "data_main/nodes.csv"

print("Loading sites.csv …")
sites = pd.read_csv(SITES_CSV, low_memory=False)
print(f"  {len(sites):,} charging sites")

# ── n_chargers: count non-null connector types per site ───────────────────────
rp_type_cols = [
    c for c in sites.columns
    if re.match(
        r"site/energyInfrastructureStation\[1\]/refillPoint\[\d+\]/connector\[1\]/connectorType\[1\]",
        c,
    )
]
sites["n_chargers"] = sites[rp_type_cols].notna().sum(axis=1)
sites["n_chargers"] = sites["n_chargers"].clip(lower=1)  # at least 1

# ── mean_power_kw: mean of maxPowerAtSocket across refillPoints (W → kW) ──────
pw_cols = [
    c for c in sites.columns
    if "refillPoint" in c and "connector[1]/maxPowerAtSocket[1]" in c
]
pw_data = sites[pw_cols].apply(pd.to_numeric, errors="coerce") / 1000.0  # W → kW
sites["mean_power_kw"] = pw_data.mean(axis=1).fillna(22.0)  # 22 kW AC fallback

print(f"  n_chargers: median={sites['n_chargers'].median():.0f}  "
      f"max={sites['n_chargers'].max():.0f}")
print(f"  mean_power_kw: median={sites['mean_power_kw'].median():.1f}  "
      f"max={sites['mean_power_kw'].max():.1f}")

# ── Load nodes ────────────────────────────────────────────────────────────────
print("\nLoading nodes.csv …")
nodes = pd.read_csv(NODES_CSV)
existing_mask = nodes["is_existing_charger"] == 1
existing = nodes[existing_mask].copy()
print(f"  {existing_mask.sum():,} existing charger nodes to enrich")

# ── Match by nearest lat/lon (should be exact 0 m for all rows) ──────────────
sites_xy = np.radians(sites[["latitude", "longitude"]].values)
existing_xy = np.radians(existing[["lat", "lon"]].values)

tree = cKDTree(sites_xy)
dists_rad, match_idx = tree.query(existing_xy, k=1)
dists_m = dists_rad * 6_371_000

print(f"  Match distances (m): "
      f"min={dists_m.min():.1f}  p50={np.median(dists_m):.1f}  "
      f"p95={np.percentile(dists_m, 95):.1f}  max={dists_m.max():.1f}")
if dists_m.max() > 1000:
    print(f"  WARNING: {(dists_m > 1000).sum()} matches > 1 km — check data quality")

# ── Write back to nodes ───────────────────────────────────────────────────────
nodes["n_chargers"]    = 0
nodes["mean_power_kw"] = 0.0

nodes.loc[existing_mask, "n_chargers"]    = sites["n_chargers"].values[match_idx]
nodes.loc[existing_mask, "mean_power_kw"] = sites["mean_power_kw"].values[match_idx]

nodes.to_csv(NODES_CSV, index=False)
print(f"\nUpdated {NODES_CSV}")
print(f"  Existing chargers — n_chargers distribution:")
ec = nodes[existing_mask]["n_chargers"].value_counts().sort_index()
for v, cnt in ec.head(10).items():
    print(f"    {v:3.0f} chargers: {cnt:,}")
print(f"  Existing chargers — mean_power_kw distribution (kW):")
bins = [0, 10, 30, 60, 100, 200, 400, 1000]
print(pd.cut(nodes[existing_mask]["mean_power_kw"], bins).value_counts().sort_index().to_string())

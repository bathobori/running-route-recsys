"""04_route_identity.py

Extracted from docs/docs/dataset.md. Edit the guide and re-extract,
or edit here and keep the guide in step.

    python src/04_route_identity.py
"""

import os, math, argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "workouts.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--cell-m", type=float, default=200)        # grid cell in metres
ap.add_argument("--dist-bucket", type=float, default=0.5)   # distance bucket in km
ap.add_argument("--save", action="store_true")              # only this writes to disk
args = ap.parse_args()

df = pd.read_csv(PATH)
DEG = args.cell_m / 111_320.0        # one degree of latitude is about 111.32 km

def cell(lat, lon):
    """Grid cell. Longitude degrees are corrected for latitude."""
    f = max(math.cos(math.radians(lat)), 0.01)
    return (round(lat / DEG), round(lon / (DEG / f)))

keys = [(cell(r.start_lat, r.start_lon),
         cell(r.end_lat, r.end_lon),
         round(r.distance_km / args.dist_bucket))
        for r in df.itertuples()]
df["route_id"] = pd.factorize(pd.Series(keys, dtype="object"))[0]

g = df.groupby("route_id").agg(n_workouts=("workout_id", "size"),
                               n_users=("user_id", "nunique"))
print(f"cell={args.cell_m} m, bucket={args.dist_bucket} km")
print(f"  routes        : {len(g)}")
print(f"  mean n_users  : {g.n_users.mean():.2f}")
print(f"  >=2 users     : {(g.n_users >= 2).sum()} routes "
      f"({100*(g.n_users >= 2).mean():.1f}%)")

if args.save:
    df.to_csv(PATH, index=False)
    print("  saved.")

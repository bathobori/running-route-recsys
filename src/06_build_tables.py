"""05_build_tables.py

Extracted from docs/docs/dataset.md. Edit the guide and re-extract,
or edit here and keep the guide in step.

    python src/05_build_tables.py
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

df = pd.read_csv(os.path.join(DATA, "workouts.csv"))
mode = lambda s: s.mode().iat[0] if len(s.mode()) else None

# --- interactions: this is what the model consumes ---
df[["user_id", "route_id", "start_ts"]] \
  .rename(columns={"start_ts": "timestamp"}) \
  .to_csv(os.path.join(DATA, "interactions.csv"), index=False)

# --- routes: the items ---
routes = df.groupby("route_id").agg(
    city=("city", mode), country=("country", mode),
    start_lat=("start_lat", "median"), start_lon=("start_lon", "median"),
    length_km=("distance_km", "median"),
    n_workouts=("workout_id", "size"), n_users=("user_id", "nunique"),
).reset_index()
routes.to_csv(os.path.join(DATA, "routes.csv"), index=False)

# --- users ---
df.groupby("user_id").agg(
    gender=("gender", "first"), n_runs=("workout_id", "size"),
    home_city=("city", mode),
).reset_index().to_csv(os.path.join(DATA, "users.csv"), index=False)

print(f"interactions : {len(df)}")
print(f"routes       : {len(routes)}")
print(f"users        : {df.user_id.nunique()}")

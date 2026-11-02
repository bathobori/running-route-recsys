"""06_analyse_sharing.py

Measures whether FitRec route data can support collaborative filtering at all.

Collaborative filtering needs items that several users interact with. FitRec stores
unique GPS traces, so an item only exists once similar traces are collapsed onto a
shared route id. This script sweeps that collapsing over grid resolutions, key
definitions and geographies, and reports how much sharing actually results.

The number that matters is **interaction coverage**: the share of workouts that fall on
a route at least two users ran. Mean users-per-route is reported too, but it is
dominated by the long tail of singleton routes.

Reads  data/workouts.csv  (produced by steps 02 and 03)
Writes sharing_analysis.json and prints a readable report.

    python src/06_analyse_sharing.py
"""

import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKOUTS = os.path.join(ROOT, "data", "workouts.csv")
OUT = os.path.join(ROOT, "results", "sharing_analysis.json")

# Rough city centres, used to test radius-based markets against city labels.
CENTRES = {
    "Copenhagen": (55.6761, 12.5683),
    "Aarhus": (56.1629, 10.2039),
    "Aalborg": (57.0488, 9.9217),
}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def grid(frame, cell_m):
    """Round start and end points onto a metric grid, correcting longitude for latitude."""
    deg = cell_m / 111_320.0

    def cell(lat, lon):
        f = max(math.cos(math.radians(lat)), 0.01)
        return (round(lat / deg), round(lon / (deg / f)))

    return ([cell(a, b) for a, b in zip(frame.start_lat, frame.start_lon)],
            [cell(a, b) for a, b in zip(frame.end_lat, frame.end_lon)])


def measure(frame, cell_m, bucket_km, key="start+end+dist"):
    """Collapse traces onto route ids and report how much sharing results."""
    starts, ends = grid(frame, cell_m)
    dist = (frame.distance_km / bucket_km).round()
    if key == "start+end+dist":
        keys = list(zip(starts, ends, dist))
    elif key == "start+dist":
        keys = list(zip(starts, dist))
    elif key == "start+end":
        keys = list(zip(starts, ends))
    elif key == "start":
        keys = starts
    else:
        raise ValueError(key)

    route_id = pd.factorize(pd.Series(keys, dtype="object"))[0]
    t = pd.DataFrame({"user": frame.user_id.values, "route": route_id})
    per_route = t.groupby("route").user.nunique()
    shared = set(per_route[per_route >= 2].index)
    return {
        "workouts": int(len(frame)),
        "users": int(frame.user_id.nunique()),
        "routes": int(len(per_route)),
        "mean_users_per_route": round(float(per_route.mean()), 3),
        "pct_routes_2plus_users": round(100 * float((per_route >= 2).mean()), 1),
        "pct_interactions_on_shared_routes": round(100 * float(t.route.isin(shared).mean()), 1),
    }


def main():
    df = pd.read_csv(WORKOUTS)
    print(f"{len(df):,} workouts, {df.user_id.nunique()} users\n")
    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "generated_by": "src/06_analyse_sharing.py",
        "source_workouts": int(len(df)),
        "source_users": int(df.user_id.nunique()),
    }

    def show(label, m):
        print(f"  {label:30} routes={m['routes']:>7,}  mean_users={m['mean_users_per_route']:5.2f}"
              f"  >=2users={m['pct_routes_2plus_users']:5.1f}%"
              f"  coverage={m['pct_interactions_on_shared_routes']:5.1f}%")

    print("GLOBAL — strict key (start + end + distance), by grid resolution")
    report["global_by_cell"] = {}
    for cell_m in [100, 200, 500, 1000, 2000]:
        m = measure(df, cell_m, 0.5)
        report["global_by_cell"][str(cell_m)] = m
        show(f"cell {cell_m} m", m)

    print("\nGLOBAL — by key definition (1000 m cells)")
    report["global_by_key"] = {}
    for key in ["start+end+dist", "start+dist", "start+end", "start"]:
        m = measure(df, 1000, 0.5, key)
        report["global_by_key"][key] = m
        show(key, m)

    print("\nGEOGRAPHIC DISPERSION")
    per_city = df.groupby("city").agg(runs=("workout_id", "size"),
                                      users=("user_id", "nunique"))
    disp = {
        "cities": int(len(per_city)),
        "cities_with_one_user": int((per_city.users == 1).sum()),
        "pct_cities_with_one_user": round(100 * float((per_city.users == 1).mean()), 1),
        "pct_runs_in_one_user_cities": round(
            100 * float(per_city[per_city.users == 1].runs.sum() / per_city.runs.sum()), 1),
        "cities_with_10plus_users": int((per_city.users >= 10).sum()),
        "pct_runs_in_10plus_user_cities": round(
            100 * float(per_city[per_city.users >= 10].runs.sum() / per_city.runs.sum()), 1),
    }
    report["dispersion"] = disp
    for k, v in disp.items():
        print(f"  {k:32} {v:,}" if isinstance(v, int) else f"  {k:32} {v}")

    print("\nTOP CITIES BY USER COUNT (1000 m cells, 1 km buckets)")
    report["top_cities"] = {}
    for city in per_city.sort_values("users", ascending=False).head(12).index:
        m = measure(df[df.city == city], 1000, 1.0)
        report["top_cities"][city] = m
        print(f"  {city[:20]:22} {m['workouts']:>6,} runs {m['users']:>4} users"
              f"  mean_users={m['mean_users_per_route']:5.2f}"
              f"  coverage={m['pct_interactions_on_shared_routes']:5.1f}%")

    print("\nRADIUS-BASED MARKETS (does merging a metro area help?)")
    report["radius_markets"] = {}
    for name, (clat, clon) in CENTRES.items():
        d = [haversine_km(clat, clon, a, b) for a, b in zip(df.start_lat, df.start_lon)]
        s = pd.Series(d, index=df.index)
        report["radius_markets"][name] = {}
        for radius in [10, 20, 35]:
            sub = df[s <= radius]
            if len(sub) < 50:
                continue
            m = measure(sub, 1000, 1.0)
            report["radius_markets"][name][f"{radius}km"] = m
            print(f"  {name[:14]:16} r={radius:>3} km  {m['workouts']:>6,} runs"
                  f" {m['users']:>4} users  coverage={m['pct_interactions_on_shared_routes']:5.1f}%")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

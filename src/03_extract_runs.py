"""02_extract_runs.py

Extracted from docs/docs/dataset.md. Edit the guide and re-extract,
or edit here and keep the guide in step.

    python src/02_extract_runs.py
"""

import os, re, ast, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(os.path.dirname(ROOT), "FitRec", "original")
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(RAW, "endomondoHR.json")

N_POINTS = 100          # resample every trajectory to this many points
MIN_DISTINCT_TS = 50    # below this the trajectory is degenerate

RE_ID = re.compile(r"'id':\s*(\d+)")
RE_USER = re.compile(r"'userId':\s*(\d+)")
RE_SPORT = re.compile(r"'sport':\s*'([^']*)'")
RE_GENDER = re.compile(r"'gender':\s*'([^']*)'")
RE_LAT = re.compile(r"'latitude':\s*(\[[^\]]*\])")
RE_LON = re.compile(r"'longitude':\s*(\[[^\]]*\])")
RE_TS = re.compile(r"'timestamp':\s*(\[[^\]]*\])")

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

# --- which workout ids survive the Meta-based filter? ---
meta = pd.read_csv(os.path.join(DATA, "meta.csv"))
ok = meta[(meta.sport == "run")
          & meta.distance_km.between(0.5, 100)
          & meta.duration_s.between(300, 21600)]      # 5 minutes .. 6 hours
meta_ix = ok.set_index("id")
allowed = set(meta_ix.index)
print(f"runs allowed by the Meta filter: {len(allowed)}")

rows, trajs = [], []
seen = 0
with open(SRC, encoding="utf-8", errors="replace") as f:
    for line in f:
        ms = RE_SPORT.search(line)
        if not ms or ms.group(1) != "run":
            continue
        mi = RE_ID.search(line)
        if not mi:
            continue
        seen += 1
        if args.limit and seen >= args.limit:
            break
        wid = int(mi.group(1))
        if wid not in allowed:
            continue
        mlat, mlon, mts = RE_LAT.search(line), RE_LON.search(line), RE_TS.search(line)
        if not (mlat and mlon and mts):
            continue
        lat = ast.literal_eval(mlat.group(1))
        lon = ast.literal_eval(mlon.group(1))
        ts = ast.literal_eval(mts.group(1))
        n = min(len(lat), len(lon), len(ts))     # lengths are not guaranteed equal
        if n < N_POINTS or len(set(ts[:n])) < MIN_DISTINCT_TS:
            continue

        idx = np.linspace(0, n - 1, N_POINTS).astype(int)
        trajs.append(np.stack([np.asarray(lat[:n])[idx],
                               np.asarray(lon[:n])[idx]], axis=1).astype("float32"))

        mu, mg = RE_USER.search(line), RE_GENDER.search(line)
        m = meta_ix.loc[wid]
        rows.append({
            "workout_id": wid,
            "user_id": int(mu.group(1)) if mu else -1,
            "gender": mg.group(1) if mg else "",
            "start_ts": int(ts[0]),
            "start_lat": float(lat[0]), "start_lon": float(lon[0]),
            "end_lat": float(lat[n - 1]), "end_lon": float(lon[n - 1]),
            "n_points": len(set(ts[:n])),
            "distance_km": float(m.distance_km),
            "duration_s": float(m.duration_s),
            "ascent": m.ascent, "descent": m.descent,
            "calories": m.calories,
            "weather_type": m.weather_type, "temperature": m.weather_temperature,
        })
        if len(rows) % 10000 == 0:
            print(f"  {len(rows)} kept ...", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(DATA, "workouts.csv"), index=False)
arr = np.stack(trajs)
np.save(os.path.join(DATA, "trajectories.npy"), arr)
print(f"DONE: {len(rows)} runs, trajectories shape={arr.shape}")

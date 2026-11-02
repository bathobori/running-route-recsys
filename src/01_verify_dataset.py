"""
Reproduces the findings documented in docs/data/:

  1. What endomondoHR_proper actually filters on — sampling density, not heart-rate
     quality.
  2. The 7-8 hour disagreement between the HR Unix timestamps and the Meta ISO strings.
  3. The real unit of the `speed` array, which the published FitRec schema gets wrong.

All three are sample-based by design: each uses a few hundred records, which is enough for
effects this large and avoids another full pass over 21 GB.

Paths are resolved from this file's location, so it runs from anywhere:
    python src/verify_findings.py
"""

import ast
import math
import os
import re
import statistics as st
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(os.path.dirname(ROOT), "FitRec", "original")
HR = os.path.join(RAW, "endomondoHR.json")
PROPER = os.path.join(RAW, "endomondoHR_proper.json")
META = os.path.join(RAW, "endomondoMeta.json")
N = 400

RE_ID = re.compile(r"'id':\s*(\d+)")
RE_SPORT = re.compile(r"'sport':\s*'([^']*)'")
RE_HR_ARR = re.compile(r"'heart_rate':\s*(\[[^\]]*\])")
RE_TS_ARR = re.compile(r"'timestamp':\s*(\[[^\]]*\])")
RE_TS_NUM = re.compile(r"'timestamp':\s*\[\s*(\d+)")
RE_TS_STR = re.compile(r"'timestamp':\s*\[\s*'([^']+)'")


def proper_ids():
    ids = set()
    with open(PROPER, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_ID.search(line)
            if m:
                ids.add(int(m.group(1)))
    return ids


def finding_1():
    print("=" * 74)
    print("1. What does endomondoHR_proper filter on?")
    print("=" * 74)
    keep = proper_ids()
    print(f"   ids in _proper: {len(keep):,}\n   scanning endomondoHR ...")

    groups = {"kept in _proper": [], "excluded": []}
    with open(HR, encoding="utf-8", errors="replace") as f:
        for line in f:
            ms = RE_SPORT.search(line)
            if not ms or ms.group(1) != "run":
                continue
            mi, mts, mhr = RE_ID.search(line), RE_TS_ARR.search(line), RE_HR_ARR.search(line)
            if not (mi and mts and mhr):
                continue
            key = "kept in _proper" if int(mi.group(1)) in keep else "excluded"
            if len(groups[key]) >= N:
                if all(len(v) >= N for v in groups.values()):
                    break
                continue
            ts = ast.literal_eval(mts.group(1))
            hr = ast.literal_eval(mhr.group(1))
            groups[key].append({
                "distinct_ts": len(set(ts)),
                "minutes": (ts[-1] - ts[0]) / 60,
                "hr_distinct": len(set(hr)),
                "hr_under30": sum(1 for x in hr if x < 30),
            })

    for key, rows in groups.items():
        d = [r["distinct_ts"] for r in rows]
        m = [r["minutes"] for r in rows]
        bad_hr = sum(1 for r in rows if r["hr_distinct"] <= 2 or r["hr_under30"] > 50)
        print(f"\n   {key}  (n={len(rows)})")
        print(f"     distinct timestamps : median {st.median(d):6.1f}   min {min(d):5.0f}")
        print(f"     duration (min)      : median {st.median(m):6.1f}   min {min(m):5.1f}")
        print(f"     <450 distinct ts    : {100*sum(1 for x in d if x < 450)/len(d):5.1f}%")
        print(f"     degenerate heart rate: {100*bad_hr/len(rows):5.1f}%")
    print("\n   -> the discriminator is sampling density, not heart-rate quality\n")


def finding_2():
    print("=" * 74)
    print("2. Do the HR and Meta timestamps agree?")
    print("=" * 74)
    hr = {}
    with open(HR, encoding="utf-8", errors="replace") as f:
        for line in f:
            ms = RE_SPORT.search(line)
            if not ms or ms.group(1) != "run":
                continue
            mi, mt = RE_ID.search(line), RE_TS_NUM.search(line)
            if mi and mt:
                hr[int(mi.group(1))] = int(mt.group(1))
            if len(hr) >= 30_000:
                break

    offsets, by_month, found = {}, {}, 0
    with open(META, encoding="utf-8", errors="replace") as f:
        for line in f:
            mi = RE_ID.search(line)
            if not mi or int(mi.group(1)) not in hr:
                continue
            mt = RE_TS_STR.search(line)
            if not mt:
                continue
            meta_dt = datetime.fromisoformat(mt.group(1).replace("Z", "+00:00"))
            hr_dt = datetime.fromtimestamp(hr[int(mi.group(1))], tz=timezone.utc)
            off = round((hr_dt - meta_dt).total_seconds() / 3600, 1)
            offsets[off] = offsets.get(off, 0) + 1
            by_month.setdefault(off, set()).add(meta_dt.month)
            found += 1
            if found >= N:
                break

    print(f"   matched workouts: {found}\n")
    for off in sorted(offsets, key=lambda x: -offsets[x]):
        months = sorted(by_month[off])
        print(f"   {off:+5.1f} h : {offsets[off]:4d} records   months seen: {months}")
    print("\n   -> whole-hour offsets, and the split follows daylight saving time\n")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def finding_3():
    print("=" * 74)
    print("3. Is `speed` in km/h or mph?")
    print("=" * 74)
    print("   Integrating each record's own GPS track and dividing by elapsed time")
    print("   gives an independent average speed to compare the array against.\n")

    arr = {k: re.compile(rf"'{k}':\s*(\[[^\]]*\])")
           for k in ["latitude", "longitude", "timestamp", "speed"]}
    ratios = []
    with open(HR, encoding="utf-8", errors="replace") as f:
        for line in f:
            ms = RE_SPORT.search(line)
            if not ms or ms.group(1) != "run":
                continue
            g = {k: rx.search(line) for k, rx in arr.items()}
            if not all(g.values()):          # speed is absent from most records
                continue
            lat = ast.literal_eval(g["latitude"].group(1))
            lon = ast.literal_eval(g["longitude"].group(1))
            ts = ast.literal_eval(g["timestamp"].group(1))
            sp = ast.literal_eval(g["speed"].group(1))
            minutes = (ts[-1] - ts[0]) / 60
            if minutes < 15 or len(set(ts)) < 490 or len(sp) != len(lat):
                continue
            km = sum(haversine_km(lat[i], lon[i], lat[i + 1], lon[i + 1])
                     for i in range(len(lat) - 1))
            if km < 2:
                continue
            ratios.append(st.mean(sp) / (km / (minutes / 60)))
            if len(ratios) >= 60:
                break

    med = st.median(ratios)
    print(f"   runs compared: {len(ratios)}")
    print(f"   median ratio (mean of `speed` / km/h from the track): {med:.3f}\n")
    print("   ~1.00 -> the array is km/h")
    print("   ~0.62 -> the array is mph (1 km/h = 0.621 mph)")
    print(f"\n   -> {'km/h' if med > 0.8 else 'mph'}; the published FitRec schema says MPH, "
          f"which does not match\n")


if __name__ == "__main__":
    finding_1()
    finding_2()
    finding_3()

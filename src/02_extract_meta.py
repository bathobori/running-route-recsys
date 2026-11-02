"""01_extract_meta.py

Extracted from docs/docs/dataset.md. Edit the guide and re-extract,
or edit here and keep the guide in step.

    python src/01_extract_meta.py
"""

import ast, os, re, csv, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(os.path.dirname(ROOT), "FitRec", "original")
DATA = os.path.join(ROOT, "data")
SRC = os.path.join(RAW, "endomondoMeta.json")
OUT = os.path.join(DATA, "meta.csv")

NUM = ["id", "userId", "distance", "duration", "calories", "ascent", "descent"]
RE_NUM = {k: re.compile(rf"'{k}':\s*([^,}}]+)") for k in NUM}
RE_SPORT = re.compile(r"'sport':\s*'([^']*)'")
RE_GENDER = re.compile(r"'gender':\s*'([^']*)'")
# weather is sometimes {'type': N} and sometimes a fuller dict with temperature,
# humidity and wind. Match any dict; a pattern ending in \} after the number
# silently skips every rich record.
RE_WEATHER = re.compile(r"'weather':\s*(\{[^}]*\})")
WEATHER_KEYS = ["type", "temperature", "humidity", "wind_speed", "wind_direction"]

def num(line, key):
    m = RE_NUM[key].search(line)
    if not m:
        return None
    s = m.group(1).strip()
    return None if s == "None" else float(s)

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

os.makedirs(DATA, exist_ok=True)
n = 0
with open(SRC, encoding="utf-8", errors="replace") as f, \
     open(OUT, "w", newline="", encoding="utf-8") as g:
    w = csv.writer(g)
    w.writerow(["id", "user_id", "sport", "gender", "distance_km", "duration_s",
                "calories", "ascent", "descent"]
               + [f"weather_{k}" for k in WEATHER_KEYS])
    for line in f:
        if not line.strip():
            continue
        wid, uid = num(line, "id"), num(line, "userId")
        if wid is None:
            continue
        ms, mg = RE_SPORT.search(line), RE_GENDER.search(line)
        mw = RE_WEATHER.search(line)
        weather = ast.literal_eval(mw.group(1)) if mw else {}
        w.writerow([
            int(wid), int(uid) if uid is not None else "",
            ms.group(1) if ms else "", mg.group(1) if mg else "",
            num(line, "distance"), num(line, "duration"),
            num(line, "calories"), num(line, "ascent"), num(line, "descent"),
        ] + [weather.get(k, "") for k in WEATHER_KEYS])
        n += 1
        if n % 100000 == 0:
            print(f"  {n} records ...", flush=True)
        if args.limit and n >= args.limit:
            break
print(f"DONE: {n} rows -> {OUT}")

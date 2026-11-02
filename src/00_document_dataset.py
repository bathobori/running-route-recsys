"""
Builds the human-readable documentation samples for the three FitRec files.

Reads each raw file exactly once and produces, under :
  samples/<name>_sample.csv       200 records spread evenly across the file
  samples/<name>_one_record.txt   one full record, pretty-printed
  samples/<name>_one_workout.csv  that same record unrolled, one row per point
  dataset_stats.json               every number quoted in the .md dossiers

The one_record and one_workout files of a given dataset describe the SAME
workout on purpose, so they read as a matched pair: the stored form, and the
same thing with its arrays laid out vertically.

Paths are resolved from this file's location, so it runs from anywhere:
    python src/build_data_docs.py
"""

import ast
import csv
import json
import os
import re
import collections
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(os.path.dirname(ROOT), "FitRec", "original")
SAMPLES = os.path.join(ROOT, "samples")
SAMPLE_ROWS = 200

# Record counts established earlier; used to spread the sample across the whole
# file in a single pass. The run asserts against them, so a changed file is loud.
FILES = {
    "endomondoHR": {"file": "endomondoHR.json", "expected": 253_020},
    "endomondoHR_proper": {"file": "endomondoHR_proper.json", "expected": 167_783},
    "endomondoMeta": {"file": "endomondoMeta.json", "expected": 962_190},
}

RE_SPORT = re.compile(r"'sport':\s*'([^']*)'")
RE_GENDER = re.compile(r"'gender':\s*'([^']*)'")
RE_ID = re.compile(r"'id':\s*(\d+)")
RE_USER = re.compile(r"'userId':\s*(\d+)")
META_SCALARS = ["distance", "duration", "calories", "ascent", "descent", "hydration"]
RE_META = {k: re.compile(rf"'{k}':\s*([^,}}]+)") for k in META_SCALARS}
# weather is sometimes {'type': N} and sometimes a fuller dict with temperature,
# humidity and wind — match any dict, or an earlier regex silently undercounts it.
RE_WEATHER = re.compile(r"'weather':\s*(\{[^}]*\})")
RE_WEATHER_RICH = re.compile(r"'weather':\s*\{[^}]*'temperature'")


def fmt(v):
    return f"{v:.6f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def summarise(v):
    """Collapse a long array into one readable cell. ASCII only, so the CSV
    opens identically in Excel, a text editor and GitHub."""
    if not isinstance(v, list):
        return v
    if not v:
        return "[0]"
    return f"[{len(v)}] {fmt(v[0])} ... {fmt(v[-1])}"


def start_utc(rec):
    ts = rec.get("timestamp")
    if not isinstance(ts, list) or not ts:
        return ""
    if isinstance(ts[0], str):          # Meta stores ISO strings
        return ts[0]
    return datetime.fromtimestamp(ts[0], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def derived(rec):
    lens = {len(v) for v in rec.values() if isinstance(v, list)}
    array_len = min(lens) if len(lens) == 1 else (
        "/".join(str(x) for x in sorted(lens)) if lens else "")
    ts = rec.get("timestamp")
    base = {"array_len": array_len, "start_utc": start_utc(rec)}
    if not isinstance(ts, list) or len(ts) < 2 or isinstance(ts[0], str):
        return {**base, "duration_min": "", "n_distinct_ts": ""}
    return {
        **base,
        "duration_min": round((ts[-1] - ts[0]) / 60, 1),
        "n_distinct_ts": len(set(ts)),
    }


def scan(name, cfg):
    path = os.path.join(RAW, cfg["file"])
    k = max(cfg["expected"] // SAMPLE_ROWS, 1)
    is_meta = name == "endomondoMeta"

    total = 0
    sports = collections.Counter()
    run_gender = collections.Counter()
    ids = set()
    run_users = set()
    meta_filled = collections.Counter()
    meta_filled_run = collections.Counter()
    speed_present = {"all": 0, "run": 0}
    sampled = []          # list of (parsed record, raw line)

    print(f"  {cfg['file']} ...", flush=True)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1

            ms = RE_SPORT.search(line)
            sport = ms.group(1) if ms else "?"
            sports[sport] += 1
            mi = RE_ID.search(line)
            if mi:
                ids.add(int(mi.group(1)))
            has_speed = "'speed':" in line
            if has_speed:
                speed_present["all"] += 1
            if sport == "run":
                mg = RE_GENDER.search(line)
                run_gender[mg.group(1) if mg else "?"] += 1
                mu = RE_USER.search(line)
                if mu:
                    run_users.add(int(mu.group(1)))
                if has_speed:
                    speed_present["run"] += 1
            if is_meta:
                target = [meta_filled] + ([meta_filled_run] if sport == "run" else [])
                for key, rx in RE_META.items():
                    m = rx.search(line)
                    if m and m.group(1).strip() != "None":
                        for c in target:
                            c[key] += 1
                if RE_WEATHER.search(line):
                    for c in target:
                        c["weather"] += 1
                    if RE_WEATHER_RICH.search(line):
                        for c in target:
                            c["weather_with_temperature"] += 1

            if (total - 1) % k == 0 and len(sampled) < SAMPLE_ROWS:
                sampled.append((ast.literal_eval(line.strip()), line))

            if total % 200_000 == 0:
                print(f"    {total:,} ...", flush=True)

    assert total == cfg["expected"], (
        f"{cfg['file']}: {total} records, expected {cfg['expected']} — "
        "the raw file changed, update FILES before trusting the docs"
    )

    recs = [r for r, _ in sampled]
    field_counts = collections.Counter(len(r) for r in recs)
    array_fields = sorted({k2 for r in recs for k2, v in r.items() if isinstance(v, list)})
    scalar_fields = sorted({k2 for r in recs for k2, v in r.items() if not isinstance(v, list)})

    write_sample_csv(name, recs, scalar_fields, array_fields)

    # one_record.txt and one_workout.csv deliberately show the SAME workout,
    # so the two files read as a matched pair: the record, and the record unrolled.
    example, raw = pick_example(sampled)
    write_one_record(name, example, raw[:600])
    write_one_workout(name, example)

    stats = {
        "file": cfg["file"],
        "size_gb": round(os.path.getsize(path) / 1024**3, 2),
        "total_records": total,
        "running_records": sports.get("run", 0),
        "running_users": len(run_users),
        "unique_ids": len(ids),
        "n_sports": len(sports),
        "sports": dict(sports.most_common()),
        "running_gender": dict(run_gender),
        "sample_field_counts": {str(a): b for a, b in sorted(field_counts.items())},
        "scalar_fields": scalar_fields,
        "array_fields": array_fields,
    }
    if not is_meta:
        stats["speed_field_present"] = {
            "records": speed_present["all"],
            "pct_of_all": round(100 * speed_present["all"] / total, 1),
            "running_records": speed_present["run"],
            "pct_of_running": round(100 * speed_present["run"] / sports["run"], 1),
        }
    lens = collections.Counter()
    for r in recs:
        for v in r.values():
            if isinstance(v, list):
                lens[len(v)] += 1
    stats["array_lengths_in_sample"] = {
        "note": f"across all array fields of the {len(recs)} sampled records",
        "exactly_500_pct": round(100 * lens[500] / max(sum(lens.values()), 1), 1),
        "top": {str(a): b for a, b in lens.most_common(8)},
    }
    if is_meta:
        # Kept separately because the dossiers quote the running-only figures,
        # and the two populations differ enough to look like a contradiction.
        fields = META_SCALARS + ["weather", "weather_with_temperature"]
        n_run = sports.get("run", 1)
        stats["field_fill_rate_pct"] = {
            "all_records": {k: round(100 * meta_filled[k] / total, 1) for k in fields},
            "running_records": {k: round(100 * meta_filled_run[k] / n_run, 1)
                                for k in fields},
        }
    stats["example_workout"] = describe_example(example)
    return stats, ids


def write_sample_csv(name, recs, scalars, arrays):
    cols = ["_row"] + scalars + arrays + ["array_len", "start_utc", "duration_min",
                                          "n_distinct_ts", "n_fields"]
    out = os.path.join(SAMPLES, f"{name}_sample.csv")
    # utf-8-sig so Excel opens the … and accented values correctly
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(recs):
            row = {"_row": i, "n_fields": len(r)}
            row.update({c: r.get(c, "") for c in scalars})
            row.update({c: summarise(r[c]) if c in r else "" for c in arrays})
            row.update(derived(r))
            w.writerow(row)
    print(f"    -> {out}")


def write_one_record(name, rec, raw):
    out = os.path.join(SAMPLES, f"{name}_one_record.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"One complete record from {name}.json\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Fields: {len(rec)}\n\n")
        for key in sorted(rec):
            v = rec[key]
            if isinstance(v, list):
                head = ", ".join(fmt(x) for x in v[:5])
                tail = ", ".join(fmt(x) for x in v[-5:])
                f.write(f"{key:12} list, {len(v)} items\n")
                f.write(f"{'':12}   first 5: [{head}]\n")
                f.write(f"{'':12}   last  5: [{tail}]\n")
            else:
                f.write(f"{key:12} {type(v).__name__}: {v!r}\n")
        f.write("\n" + "=" * 72 + "\n")
        f.write("First 600 characters of the RAW line.\n")
        f.write("Note the single quotes and Python None: this is a Python dict repr,\n")
        f.write("not JSON. Parse it with ast.literal_eval, never json.loads.\n")
        f.write("=" * 72 + "\n\n")
        f.write(raw + "\n")
    print(f"    -> {out}")


def pick_example(sampled):
    """Choose the record that one_record.txt and one_workout.csv both describe.

    Scored rather than filtered, so there is always a winner. We want a run,
    with every field present, a dense series, and as few empty values as
    possible — a sample record that is quietly broken (a 52-minute run with
    distance 0.0, say) teaches the reader the wrong thing.
    """
    widest = max((len(r) for r, _ in sampled), default=0)

    def score(rec):
        ts = rec.get("timestamp") or []
        scalars = [v for v in rec.values() if not isinstance(v, list)]
        distance = rec.get("distance")
        return (
            rec.get("sport") == "run",
            len(rec) == widest,
            distance is None or distance > 0,      # no effect where absent
            len(set(ts)) >= 490,
            sum(1 for v in scalars if v is not None),
        )

    return max(sampled, key=lambda pair: score(pair[0]))


def describe_example(rec):
    ts = rec.get("timestamp") or []
    return {
        "workout_id": rec.get("id"),
        "user_id": rec.get("userId"),
        "sport": rec.get("sport"),
        "points": len(ts),
        "start_utc": start_utc(rec),
        "url": rec.get("url"),
    }


def write_one_workout(name, rec):
    """The same record, unrolled: one row per recorded point.

    Only the array fields vary per point. The scalars describe the whole
    workout, but they are repeated on every row anyway so that the file stands
    on its own — opened alone, it still says which workout and which sport.
    """
    arrays = sorted(k for k, v in rec.items() if isinstance(v, list))
    scalars = sorted(k for k, v in rec.items() if not isinstance(v, list))
    if not arrays:
        print(f"    !! {name}: no array fields, skipping one_workout")
        return
    n = min(len(rec[k]) for k in arrays)

    ts = rec.get("timestamp")
    numeric_ts = isinstance(ts, list) and ts and not isinstance(ts[0], str)

    cols = scalars + ["point_idx"] + (["utc_time"] if numeric_ts else []) + arrays
    out = os.path.join(SAMPLES, f"{name}_one_workout.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            row = [rec[k] for k in scalars] + [i]
            if numeric_ts:
                row.append(datetime.fromtimestamp(ts[i], tz=timezone.utc)
                           .strftime("%Y-%m-%d %H:%M:%S"))
            row += [rec[k][i] for k in arrays]
            w.writerow(row)
    print(f"    -> {out}  ({n} rows)")


def main():
    os.makedirs(SAMPLES, exist_ok=True)
    print("Scanning raw files (one pass each) ...")

    results, id_sets = {}, {}
    for name, cfg in FILES.items():
        results[name], id_sets[name] = scan(name, cfg)

    names = list(FILES)
    overlaps = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = id_sets[names[i]], id_sets[names[j]]
            overlaps[f"{names[i]} & {names[j]}"] = {
                "common": len(a & b),
                f"only_{names[i]}": len(a - b),
                f"only_{names[j]}": len(b - a),
            }

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "generated_by": "src/build_data_docs.py",
        "sample_rows_per_file": SAMPLE_ROWS,
        "files": results,
        "id_overlaps": overlaps,
    }
    out = os.path.join(ROOT, "results", "dataset_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n-> {out}")

    for name, r in results.items():
        print(f"{name:22} {r['total_records']:>9,} records, "
              f"{r['running_records']:>8,} runs, {r['running_users']:>5} run users")
    for pair, o in overlaps.items():
        print(f"{pair}: {o}")


if __name__ == "__main__":
    main()

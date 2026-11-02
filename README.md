# running-route-recsys

Working repository for a master's thesis on **running route recommendation using the
FitRec / Endomondo dataset**. It holds the code, the processed data, the measurements and
the figures that support the written thesis.

**RQ1** — how does pure collaborative filtering perform on large-scale implicit sports data?
**RQ2** — is the implicit trajectory signal sufficient without physiological telemetry?

---

## What has been established

The baseline nobody in this literature had established has now been measured end to end, on
108,853 clean running workouts from 854 users.

| | |
|---|---|
| Clean running workouts | 108,853 from 854 users |
| Routes run by exactly one person | 96.6% |
| Workouts on a route ≥2 users ran | 7.3% at a 1 km grid, 11.8% at 2 km |
| Cities with a single user | 76% of 4,951 |
| Best model on the densest market | MostPopular, ahead of BPR, ALS, SVD, ItemKNN |
| Share of test cases reachable at all | 67% in Copenhagen |

Whole-route matching has a hard ceiling, and the ceiling is now quantified rather than
suspected. The natural refinement is a **smaller item** — route segments via OpenStreetMap
map-matching — so that partial overlap between runners counts. See
[docs/findings.md](docs/findings.md).

---

## Layout

```
src/        every step, numbered in the order it runs
results/    measured numbers as JSON, and 18 figures generated from them
samples/    openable extracts of the raw FitRec files
data/       processed tables — generated, not tracked
docs/       short notes: dataset reference, findings, figure catalogue
```

| | |
|---|---|
| [docs/figures.md](docs/figures.md) | **what each of the 18 figures shows** — start here when writing |
| [docs/findings.md](docs/findings.md) | the results and what they mean |
| [docs/dataset.md](docs/dataset.md) | field lists, counts and known defects of the raw data |

## Running it

Raw FitRec files (~21 GB) are not in the repository; they sit beside it at
`../FitRec/original/`.

```bash
pip install pandas numpy scipy scikit-learn matplotlib reverse_geocoder implicit
python src/02_extract_meta.py      # then 03 .. 09, in order
```

| Script | Does |
|---|---|
| `00_document_dataset.py` | samples and counts from the raw files |
| `01_verify_dataset.py` | proves three non-obvious claims about the raw data |
| `02_extract_meta.py` | 9.9 GB → `meta.csv`, workout summaries |
| `03_extract_runs.py` | 6.1 GB → `workouts.csv` + `trajectories.npy` |
| `04_geocode.py` | adds city and country |
| `05_route_identity.py` | assigns `route_id` (writes only with `--save`) |
| `06_build_tables.py` | `interactions.csv`, `routes.csv`, `users.csv` |
| `07_analyse_sharing.py` | sweeps route identity → `sharing_analysis.json` |
| `08_benchmark_cf.py` | six models, leave-one-out → `cf_benchmark.json` |
| `09_figures.py` | all 18 figures |

Scripts resolve their own paths, so they run from any directory. Steps 02 and 03 take
`--limit N` for a quick test. Every figure is generated from the files in `results/` — none
is drawn from hand-typed numbers.

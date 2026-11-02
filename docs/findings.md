# Findings

Reproduce with `src/07_analyse_sharing.py` and `src/08_benchmark_cf.py`; numbers in
[results/](../results/). Figures listed in [figures.md](figures.md).

---

## What was established

The running-route literature works either on very small samples with explicit ratings
(11–34 runners) or jumps to heart-rate-driven neural models. **The plain collaborative
filtering baseline had never been measured on a large implicit geographic dataset.** It has
now been, end to end, and the limit is quantified.

## 1 — Route identity is the binding constraint

FitRec stores unique GPS traces, not routes, so similar traces have to be collapsed onto a
shared route id before collaborative filtering is possible. Swept over five grid resolutions
and four key definitions:

| Grid cell | Routes | Mean users/route | Interaction coverage |
|---|---|---|---|
| 100 m | 75,591 | 1.01 | 3.0% |
| 200 m | 66,619 | 1.02 | 3.7% |
| 500 m | 57,034 | 1.03 | 5.2% |
| 1000 m | 51,602 | 1.04 | 7.3% |
| 2000 m | 45,786 | 1.07 | 11.8% |

**Twentyfold coarsening moves the mean from 1.01 to 1.07.** At 2 km the cell is larger than
most of the parks people run in, so the parameter space is exhausted. 96.6% of routes were
run by exactly one person.

Loosening what must match helps but costs meaning — at "same start cell only" coverage
reaches 25.2%, but the item is then a neighbourhood, not a route.

## 2 — The cause is geographic dispersion

| | |
|---|---|
| Distinct cities | 4,951 |
| Cities with exactly one user | **3,743 (75.6%)** |
| Share of all runs in those cities | 37.4% |
| Cities with ≥10 users | 67 |

Three quarters of the cities hold a single person's training log. Those runs cannot
contribute to collaborative filtering even in principle.

Ranking cities by run count misleads: Prešov has 1,465 runs from one user. The ranking that
matters is by user count, which puts **Copenhagen (89 users) ahead of Aarhus (47)** — Aarhus
had been assumed to have around 100.

Merging districts into a wider metro area makes it worse: a larger radius gathers users but
gathers area faster, so coverage falls from 36.4% at 10 km to 26.4% at 35 km.

## 3 — On the densest market, counting wins

Copenhagen, 61 evaluable users, leave-one-out, full ranking:

| Model | HR@10 | NDCG@10 |
|---|---|---|
| **MostPopular** | **0.377** | 0.218 |
| BPR | 0.328 | 0.181 |
| ALS | 0.115 | 0.044 |
| SVD | 0.082 | 0.031 |
| ItemKNN | 0.033 | 0.019 |
| Random | 0.016 | 0.016 |

Three things follow.

**The popularity baseline wins** — in four of six configurations tested, and the two where BPR
leads are within one or two users at this sample size. Note that MostPopular applies the same
score row to every user: no personalisation at all.

**SVD, the originally planned method, reaches about a fifth of the baseline.** A direct,
measured case for not using it.

**BPR is four times better than SVD** and is the best of the collaborative filtering models.
Preferring it over SVD was the right call; it still does not beat counting.

## 4 — The ceiling explains why

Only **67%** of Copenhagen test cases are routes that appear anywhere in the training data. A
third of them could not have been retrieved by any algorithm — there is nothing to learn them
from. In the widest market tested the ceiling is 44%.

**Most of the gap between the models and perfect accuracy is absent data, not model error.**
That reframes the result: the algorithms are not underperforming, they are operating against a
hard upper bound.

> One caveat worth stating explicitly: coarser grids improve every metric, because more
> genuinely different runs get declared the same route. The apparent gain is the metric
> rewarding a loss of precision, so grid size should not be tuned for better numbers.

---

## Where this goes next

Two causes combine, and only one is open to change.

| Cause | Changeable |
|---|---|
| Runners are spread across 4,951 cities | No — this is the data |
| **Whole-route matching is all-or-nothing** | **Yes — this was a design choice** |

Two people running the same stretch of riverside path from different streets, over different
distances, currently share nothing. `b8_overlaid_routes` shows this directly: Copenhagen runs
form clearly shared corridors, while complete end-to-end routes rarely coincide.

**The natural refinement is a smaller item.** Map-matching the traces onto the OpenStreetMap
network and treating each traversed **segment** as the item makes partial overlap count. Since
a segment occurs far more often than a whole route, it raises the ceiling directly — the one
quantity bounding everything above.

OSM already stores the network split at intersections, each segment carrying a stable id and
tags such as `lit`, `surface` and `highway`. Those tags are also what a safety- or
comfort-oriented extension would need, which makes this the one direction that serves RQ2 as
well.

It introduces a design question of its own: if the item is a segment, a recommendation still
has to be a runnable route, so reconstructing one from scored segments is an open step.

Pipeline steps 05 and 06 are deliberately not finalised until this is settled — committing to
a route definition already shown to have a low ceiling would be premature.

## Known limitations

**Test sets are small**, 61 to 104 users, so only large gaps are meaningful. MostPopular over
SVD is large; BPR over MostPopular in two configurations is one or two users.

**No hyperparameter tuning** — partly deliberate, since RQ1 asks for an un-optimised
benchmark, but the gap is wide enough that tuning would not close it.

**A single split, no confidence intervals.** Bootstrapping or several seeds would quantify the
uncertainty.

**The sample is ~92% male**, so it cannot support conclusions about women runners.

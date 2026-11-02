"""07_cf_benchmark.py

Does collaborative filtering beat a popularity baseline on a restricted market?

06_analyse_sharing.py showed that globally almost no two users share a route. The obvious
fallback is to restrict the study to the one place dense enough to try — Copenhagen. This
script runs that fallback end to end so the answer is measured rather than argued.

Protocol
  items        routes from the grid key (start cell, end cell, distance bucket)
  interactions unique (user, route) pairs, earliest timestamp kept
  split        leave-one-out: each user's most recent pair is the test case
  evaluation   full ranking over every route the user has not already run
  metrics      HR@10 and NDCG@10

It also reports the **ceiling**: the share of test items that appear anywhere in the
training data. A test item nobody else ever ran cannot be retrieved by any algorithm, so
no model can score above this. It is usually the most informative number here.

Reads  data/workouts.csv
Writes cf_benchmark.json

    python src/07_cf_benchmark.py
"""

import json
import math
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKOUTS = os.path.join(ROOT, "data", "workouts.csv")
OUT = os.path.join(ROOT, "results", "cf_benchmark.json")

K = 10
SEED = 42
CPH = (55.6761, 12.5683)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def route_ids(frame, cell_m, bucket_km):
    deg = cell_m / 111_320.0

    def cell(lat, lon):
        f = max(math.cos(math.radians(lat)), 0.01)
        return (round(lat / deg), round(lon / (deg / f)))

    keys = list(zip([cell(a, b) for a, b in zip(frame.start_lat, frame.start_lon)],
                    [cell(a, b) for a, b in zip(frame.end_lat, frame.end_lon)],
                    (frame.distance_km / bucket_km).round()))
    return pd.factorize(pd.Series(keys, dtype="object"))[0]


def build_split(frame, cell_m, bucket_km):
    """Unique (user, route) pairs, each user's latest held out for testing."""
    f = frame.copy()
    f["route"] = route_ids(f, cell_m, bucket_km)
    pairs = (f.groupby(["user_id", "route"], as_index=False).start_ts.min()
             .sort_values(["user_id", "start_ts"]))
    counts = pairs.groupby("user_id").size()
    eligible = set(counts[counts >= 2].index)
    pairs = pairs[pairs.user_id.isin(eligible)]

    test = pairs.groupby("user_id").tail(1)
    train = pairs.drop(test.index)

    users = {u: i for i, u in enumerate(sorted(pairs.user_id.unique()))}
    items = {r: i for i, r in enumerate(sorted(pairs.route.unique()))}
    X = sp.csr_matrix(
        (np.ones(len(train)),
         ([users[u] for u in train.user_id], [items[r] for r in train.route])),
        shape=(len(users), len(items)),
    )
    test_item = {users[u]: items[r] for u, r in zip(test.user_id, test.route)}
    return X, test_item, users, items, len(pairs)


def evaluate(scores, X, test_item):
    """HR@K and NDCG@K with the user's training items masked out."""
    hits, ndcg = 0, 0.0
    for u, t in test_item.items():
        s = scores[u].astype(float).copy()
        s[X[u].indices] = -np.inf          # never recommend what they already ran
        top = np.argpartition(-s, min(K, len(s) - 1))[:K]
        top = top[np.argsort(-s[top])]
        where = np.where(top == t)[0]
        if len(where):
            hits += 1
            ndcg += 1.0 / math.log2(where[0] + 2)
    n = len(test_item)
    return {"HR@10": round(hits / n, 4), "NDCG@10": round(ndcg / n, 4)}


def model_scores(X, rng):
    """Score matrices for every model, users x items."""
    n_users, n_items = X.shape
    out = {}

    out["Random"] = rng.random((n_users, n_items))

    pop = np.asarray(X.sum(axis=0)).ravel()
    out["MostPopular"] = np.tile(pop, (n_users, 1))

    # ItemKNN, cosine similarity between item columns
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    Xn = X.multiply(sp.csr_matrix(1.0 / norms))
    sim = (Xn.T @ Xn).toarray()
    np.fill_diagonal(sim, 0.0)
    out["ItemKNN"] = X @ sim

    factors = max(2, min(32, n_items // 10))
    try:
        from implicit.als import AlternatingLeastSquares
        from implicit.bpr import BayesianPersonalizedRanking
        als = AlternatingLeastSquares(factors=factors, regularization=0.05,
                                      iterations=30, random_state=SEED)
        als.fit(X, show_progress=False)
        out["ALS"] = np.asarray(als.user_factors) @ np.asarray(als.item_factors).T

        bpr = BayesianPersonalizedRanking(factors=factors, iterations=120,
                                          random_state=SEED)
        bpr.fit(X, show_progress=False)
        out["BPR"] = np.asarray(bpr.user_factors) @ np.asarray(bpr.item_factors).T
    except Exception as exc:                      # library missing or failed
        print(f"    (implicit models skipped: {type(exc).__name__}: {exc})")

    # Pure SVD — the method the thesis presentation proposes
    try:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(factors, min(X.shape) - 1),
                           random_state=SEED)
        out["SVD"] = svd.inverse_transform(svd.fit_transform(X))
    except Exception as exc:
        print(f"    (SVD skipped: {type(exc).__name__}: {exc})")

    return out


def main():
    df = pd.read_csv(WORKOUTS)
    dist = pd.Series([haversine_km(*CPH, a, b)
                      for a, b in zip(df.start_lat, df.start_lon)], index=df.index)

    markets = {
        "Copenhagen city label": df[df.city == "Copenhagen"],
        "Copenhagen 10 km radius": df[dist <= 10],
        "Copenhagen 20 km radius": df[dist <= 20],
    }

    rng = np.random.default_rng(SEED)
    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "generated_by": "src/07_cf_benchmark.py",
        "protocol": {
            "split": "leave-one-out on each user's most recent unique (user, route) pair",
            "ranking": "full ranking over all routes the user has not already run",
            "k": K,
        },
        "markets": {},
    }

    for label, frame in markets.items():
        for cell_m in [1000, 2000]:
            X, test_item, users, items, n_pairs = build_split(frame, cell_m, 1.0)
            if len(test_item) < 20:
                continue
            in_train = np.asarray(X.sum(axis=0)).ravel() > 0
            ceiling = float(np.mean([in_train[t] for t in test_item.values()]))

            key = f"{label} / {cell_m} m cells"
            print(f"\n{key}")
            print(f"  {X.shape[0]} users, {X.shape[1]} routes, {n_pairs} unique pairs, "
                  f"{len(test_item)} test cases")
            print(f"  CEILING (test item seen in training): {100*ceiling:.1f}%"
                  f"  <- no model can beat this")

            entry = {
                "users": int(X.shape[0]), "routes": int(X.shape[1]),
                "unique_pairs": int(n_pairs), "test_cases": int(len(test_item)),
                "ceiling_HR": round(ceiling, 4), "results": {},
            }
            for name, scores in model_scores(X, rng).items():
                m = evaluate(np.asarray(scores), X, test_item)
                entry["results"][name] = m
                print(f"    {name:14} HR@10={m['HR@10']:.4f}  NDCG@10={m['NDCG@10']:.4f}")
            report["markets"][key] = entry

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

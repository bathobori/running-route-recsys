"""09_figures.py

Renders every figure in results/figures/.

Figures are generated from the committed result files and the processed tables, never from
numbers typed by hand, so a figure cannot disagree with the analysis it illustrates.

Groups
  a*  what the dataset looks like
  b*  route identity and how much sharing it produces
  c*  model benchmark against baselines

Reads  results/sharing_analysis.json, results/cf_benchmark.json,
       data/workouts.csv, data/trajectories.npy, samples/endomondoHR_one_workout.csv
Writes results/figures/*.png

    python src/09_figures.py
"""

import json
import math
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "results", "figures")

# Validated palette: chart chrome plus two categorical slots.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"
DIM = "#c9d7ea"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("blues", RAMP)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 11, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False,
    })


def finish(ax, title, subtitle=None, xlabel=None, ylabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=13.5, fontweight="semibold",
                 loc="left", pad=30 if subtitle else 12)
    if subtitle:
        # offset in points, not axes fraction, so it clears the title at any figure size
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    color=INK_2, fontsize=10.5, va="bottom", ha="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, labelpad=8)


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> results/figures/{name}")


def route_ids(frame, cell_m, bucket_km):
    deg = cell_m / 111_320.0

    def cell(lat, lon):
        f = max(math.cos(math.radians(lat)), 0.01)
        return (round(lat / deg), round(lon / (deg / f)))

    keys = list(zip([cell(a, b) for a, b in zip(frame.start_lat, frame.start_lon)],
                    [cell(a, b) for a, b in zip(frame.end_lat, frame.end_lon)],
                    (frame.distance_km / bucket_km).round()))
    return pd.factorize(pd.Series(keys, dtype="object"))[0]


# ============================================================ a — the dataset

def a1_countries_gender(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    top = df.country.value_counts().head(8)[::-1]
    ax1.barh(list(top.index), list(top.values), color=DIM, height=0.68)
    ax1.barh([top.index[-1]], [top.values[-1]], color=S1, height=0.68)
    for i, v in enumerate(top.values):
        ax1.text(v + top.values.max() * 0.02, i, f"{v:,}", va="center", color=INK, fontsize=10)
    ax1.set_xlim(0, top.values.max() * 1.2)
    ax1.grid(axis="y", visible=False); ax1.set_xticks([])
    ax1.spines["bottom"].set_visible(False)
    finish(ax1, "Runs by country", "Top 8 countries")

    g = df.gender.fillna("unknown").value_counts()
    g = g.reindex([k for k in ["male", "female", "unknown"] if k in g.index])
    bars = ax2.bar([k.capitalize() for k in g.index], list(g.values),
                   color=[S1, S2, DIM][:len(g)], width=0.5)
    for bar, v in zip(bars, g.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + g.max() * 0.02,
                 f"{v:,}\n{100*v/g.sum():.1f}%", ha="center", va="bottom",
                 color=INK, fontsize=10.5, linespacing=1.4)
    ax2.set_ylim(0, g.max() * 1.28); ax2.set_yticks([])
    ax2.grid(axis="x", visible=False); ax2.spines["left"].set_visible(False)
    finish(ax2, "Runs by recorded gender", "All running workouts")
    fig.subplots_adjust(wspace=0.25)
    save(fig, "a1_countries_gender.png")


def a2_distance_duration(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    d = df.distance_km[df.distance_km < 30]
    ax1.hist(d, bins=60, color=S1, edgecolor=SURFACE, linewidth=0.5)
    ax1.axvline(df.distance_km.median(), color=S2, linewidth=1.6)
    ax1.text(df.distance_km.median() + 0.4, ax1.get_ylim()[1] * 0.92,
             f"median {df.distance_km.median():.1f} km", color=S2, fontsize=10.5)
    ax1.set_yticks([]); ax1.grid(axis="x", visible=False)
    ax1.spines["left"].set_visible(False)
    finish(ax1, "Distance distribution", "Runs under 30 km · 98% of the data",
           xlabel="Distance (km)")

    m = (df.duration_s / 60)[df.duration_s / 60 < 150]
    ax2.hist(m, bins=60, color=S1, edgecolor=SURFACE, linewidth=0.5)
    med = (df.duration_s / 60).median()
    ax2.axvline(med, color=S2, linewidth=1.6)
    ax2.text(med + 2, ax2.get_ylim()[1] * 0.92, f"median {med:.0f} min",
             color=S2, fontsize=10.5)
    ax2.set_yticks([]); ax2.grid(axis="x", visible=False)
    ax2.spines["left"].set_visible(False)
    finish(ax2, "Duration distribution", "Runs under 150 minutes",
           xlabel="Duration (minutes)")
    fig.subplots_adjust(wspace=0.2)
    save(fig, "a2_distance_duration.png")


def a3_runs_per_user(df):
    per_user = df.groupby("user_id").size().sort_values(ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(range(1, len(per_user) + 1), per_user.values, color=S1, linewidth=2)
    ax1.set_yscale("log")
    ax1.grid(axis="x", visible=False)
    finish(ax1, "Runs per user, ranked", f"{len(per_user)} users, descending",
           xlabel="User rank", ylabel="Runs (log scale)")

    ax2.hist(per_user.values[per_user.values < 600], bins=50, color=S1,
             edgecolor=SURFACE, linewidth=0.5)
    ax2.axvline(per_user.median(), color=S2, linewidth=1.6)
    ax2.text(per_user.median() + 12, ax2.get_ylim()[1] * 0.9,
             f"median {per_user.median():.0f} runs", color=S2, fontsize=10.5)
    ax2.set_yticks([]); ax2.grid(axis="x", visible=False)
    ax2.spines["left"].set_visible(False)
    finish(ax2, "Runs per user, distribution", "Users with fewer than 600 runs",
           xlabel="Runs per user")
    fig.subplots_adjust(wspace=0.22)
    save(fig, "a3_runs_per_user.png")


def a4_temporal(df):
    t = pd.to_datetime(df.start_ts, unit="s")
    by_month = t.dt.to_period("M").value_counts().sort_index()
    by_month = by_month[by_month.index >= pd.Period("2011-01")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [2, 1]})
    ax1.fill_between(range(len(by_month)), by_month.values, color=S1, alpha=0.25)
    ax1.plot(range(len(by_month)), by_month.values, color=S1, linewidth=1.8)
    ticks = [i for i, p in enumerate(by_month.index) if p.month == 1]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(by_month.index[i].year) for i in ticks])
    ax1.grid(axis="x", visible=False)
    finish(ax1, "Running workouts per month", "2011-2016",
           ylabel="Workouts")

    by_m = t.dt.month.value_counts().sort_index()
    names = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    peak = by_m.idxmax()
    ax2.bar(range(1, 13), by_m.reindex(range(1, 13)).values,
            color=[S2 if i == peak else DIM for i in range(1, 13)], width=0.65)
    ax2.set_xticks(range(1, 13)); ax2.set_xticklabels(names)
    ax2.set_yticks([]); ax2.grid(axis="x", visible=False)
    ax2.spines["left"].set_visible(False)
    finish(ax2, "Runs by calendar month", "All years combined")
    fig.subplots_adjust(wspace=0.2)
    save(fig, "a4_temporal_coverage.png")


def a5_example_trajectory():
    path = os.path.join(ROOT, "samples", "endomondoHR_one_workout.csv")
    w = pd.read_csv(path, encoding="utf-8-sig")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    ax1.plot(w.longitude, w.latitude, color=S1, linewidth=1.8)
    ax1.scatter([w.longitude.iloc[0]], [w.latitude.iloc[0]], s=70, color=S2,
                zorder=5, edgecolor=SURFACE, linewidth=1.5)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.grid(visible=False)
    for s in ax1.spines.values():
        s.set_visible(False)
    finish(ax1, "One workout as a path", "Workout 27247193 · orange marks the start")

    ax2.plot(w.point_idx, w.heart_rate, color=S2, linewidth=1.6, label="Heart rate (bpm)")
    ax2.plot(w.point_idx, w.altitude, color=S1, linewidth=1.6, label="Altitude (m)")
    ax2.legend(loc="upper right", fontsize=10, labelcolor=INK_2)
    ax2.grid(axis="x", visible=False)
    finish(ax2, "The same workout as series", "Each recorded point in order",
           xlabel="Recorded point")
    fig.subplots_adjust(wspace=0.15)
    save(fig, "a5_example_trajectory.png")


def a6_start_points(df):
    """Where the usable data is. Latitude/longitude needs an aspect correction or
    the map comes out stretched east-west."""
    def geo(ax, sub, lon_lim, lat_lim, colour, size, alpha):
        ax.scatter(sub.start_lon, sub.start_lat, s=size, color=colour,
                   alpha=alpha, linewidths=0)
        ax.set_xlim(*lon_lim); ax.set_ylim(*lat_lim)
        mid = (lat_lim[0] + lat_lim[1]) / 2
        ax.set_aspect(1 / math.cos(math.radians(mid)))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(visible=False)
        for s in ax.spines.values():
            s.set_color(AXIS)

    dk_lon, dk_lat = (7.8, 13.2), (54.5, 57.9)
    cp_lon, cp_lat = (12.42, 12.68), (55.60, 55.76)
    dk = df[df.start_lon.between(*dk_lon) & df.start_lat.between(*dk_lat)]
    cp = df[df.start_lon.between(*cp_lon) & df.start_lat.between(*cp_lat)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4))
    geo(ax1, dk, dk_lon, dk_lat, S1, 2, 0.20)
    finish(ax1, "Denmark", f"{len(dk):,} run start points")
    geo(ax2, cp, cp_lon, cp_lat, S2, 7, 0.30)
    finish(ax2, "Copenhagen", f"{len(cp):,} run start points")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "a6_start_points.png")


# ================================================ b — route identity and sharing

def b1_coverage_vs_grid(sharing):
    cells = sorted(sharing["global_by_cell"], key=int)
    cov = [sharing["global_by_cell"][c]["pct_interactions_on_shared_routes"] for c in cells]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(range(len(cells)), cov, color=S1, linewidth=2, marker="o", markersize=8,
            markerfacecolor=S1, markeredgecolor=SURFACE, markeredgewidth=2)
    for i in (0, len(cells) - 1):
        ax.annotate(f"{cov[i]:.1f}%", (i, cov[i]), textcoords="offset points",
                    xytext=(0, 14), ha="center", color=INK, fontsize=11,
                    fontweight="semibold")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{c} m" for c in cells])
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.text(len(cells) - 1, 72, "3.0% at 100 m\n11.8% at 2000 m", ha="right",
            va="top", color=INK_2, fontsize=11, linespacing=1.5)
    finish(ax, "Interaction coverage by grid cell size",
           "Share of workouts on a route at least two users ran",
           xlabel="Grid cell size used to decide that two traces are the same route")
    save(fig, "b1_coverage_vs_grid.png")


def b2_users_per_route(df):
    f = df.copy()
    f["route"] = route_ids(f, 1000, 0.5)
    per_route = f.groupby("route").user_id.nunique()
    labels = ["1 user", "2", "3", "4 or more"]
    vals = [int((per_route == 1).sum()), int((per_route == 2).sum()),
            int((per_route == 3).sum()), int((per_route >= 4).sum())]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(labels, vals, color=[S2] + [DIM] * 3, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.015,
                f"{v:,}\n({100*v/total:.1f}%)", ha="center", va="bottom",
                color=INK, fontsize=10.5, linespacing=1.4)
    ax.set_ylim(0, max(vals) * 1.3); ax.set_yticks([])
    ax.grid(axis="x", visible=False); ax.spines["left"].set_visible(False)
    finish(ax, "Distinct users per route",
           f"{total:,} routes worldwide · 1 km grid cells, 0.5 km distance buckets",
           xlabel="Distinct users who ran the route")
    save(fig, "b2_users_per_route.png")


def b3_coverage_by_key(sharing):
    order = ["start+end+dist", "start+dist", "start+end", "start"]
    pretty = ["start + end\n+ distance", "start\n+ distance", "start + end", "start only"]
    cov = [sharing["global_by_key"][k]["pct_interactions_on_shared_routes"] for k in order]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = [S1, S1, S1, S2]
    bars = ax.bar(pretty, cov, color=colors, width=0.55)
    for bar, v in zip(bars, cov):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.1f}%", ha="center", color=INK, fontsize=11)
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.text(3, 40, "any two runs starting in\nthe same 1 km cell",
            ha="center", va="top", color=S2, fontsize=10, linespacing=1.5)
    finish(ax, "Interaction coverage by matching rule",
           "1 km grid cells · what has to match for two runs to be the same route",
           ylabel="Coverage")
    save(fig, "b3_coverage_by_key.png")


def b4_city_dispersion(df):
    per_city = df.groupby("city").user_id.nunique()
    labels = ["1 user", "2–4", "5–9", "10 or more"]
    vals = [int((per_city == 1).sum()), int(per_city.between(2, 4).sum()),
            int(per_city.between(5, 9).sum()), int((per_city >= 10).sum())]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.barh(labels[::-1], vals[::-1], color=([S2] + [DIM] * 3)[::-1], height=0.62)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + total * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{v:,}  ({100*v/total:.0f}%)", va="center", color=INK, fontsize=11)
    ax.set_xlim(0, max(vals) * 1.22); ax.set_xticks([])
    ax.grid(axis="y", visible=False); ax.spines["bottom"].set_visible(False)
    finish(ax, "Cities by number of distinct users",
           f"{total:,} cities with at least one recorded run",
           ylabel="Distinct users in the city")
    save(fig, "b4_city_dispersion.png")


def b5_top_cities(df):
    per_city = df.groupby("city").agg(runs=("workout_id", "size"),
                                      users=("user_id", "nunique"))
    top = per_city.sort_values("users", ascending=False).head(10)[::-1]
    colors = [S2 if c == "Copenhagen" else DIM for c in top.index]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bars = ax.barh(list(top.index), list(top.users), color=colors, height=0.68)
    for bar, (name, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{int(row.users)}", va="center", color=INK, fontsize=11,
                fontweight="semibold" if name == "Copenhagen" else "normal")
    ax.set_xlim(0, top.users.max() * 1.18)
    ax.grid(axis="y", visible=False)
    finish(ax, "Top cities by distinct users",
           "Ranked by users, not by run count", xlabel="Distinct users")
    fig.text(0.5, -0.04, "Not shown: Presov, 1,465 runs from 1 user; "
             "Kecskemet, 1,391 runs from 3 users.",
             ha="center", va="top", color=INK_2, fontsize=10)
    save(fig, "b5_top_cities.png")


def b6_users_vs_coverage(sharing):
    cities = sharing["top_cities"]
    users = [v["users"] for v in cities.values()]
    cov = [v["pct_interactions_on_shared_routes"] for v in cities.values()]
    names = list(cities)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(users, cov, s=90, color=S1, alpha=0.8, edgecolor=SURFACE, linewidth=1.5)
    for n, u, c in zip(names, users, cov):
        if n in ("Copenhagen", "Arhus", "Aalborg", "Tiergarten Bezirk", "Riga"):
            ax.annotate(n, (u, c), textcoords="offset points", xytext=(8, 4),
                        color=INK_2, fontsize=10)
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    finish(ax, "Interaction coverage against city user count",
           "Each point is a city · 1 km grid cells",
           xlabel="Distinct users in the city", ylabel="Interaction coverage")
    save(fig, "b6_users_vs_coverage.png")


def b7_radius_effect(sharing):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"Copenhagen": S1, "Aarhus": S2, "Aalborg": MUTED}
    for name, entry in sharing["radius_markets"].items():
        radii = [int(k.replace("km", "")) for k in entry]
        cov = [v["pct_interactions_on_shared_routes"] for v in entry.values()]
        ax.plot(radii, cov, marker="o", markersize=8, linewidth=2,
                color=colors.get(name, MUTED), label=name,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_ylim(0, 60)
    ax.set_xticks([10, 20, 35])
    ax.set_xticklabels(["10 km", "20 km", "35 km"])
    ax.set_yticks([0, 15, 30, 45, 60])
    ax.set_yticklabels(["0%", "15%", "30%", "45%", "60%"])
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=10.5, labelcolor=INK_2)
    finish(ax, "Interaction coverage by market radius",
           "Runs within a given distance of each city centre · 1 km cells",
           xlabel="Radius around the city centre", ylabel="Interaction coverage")
    save(fig, "b7_radius_effect.png")


def b8_overlaid_routes(df):
    """Copenhagen trajectories drawn on top of each other."""
    traj = np.load(os.path.join(ROOT, "data", "trajectories.npy"), mmap_mode="r")
    idx = df.index[(df.city == "Copenhagen")][:400]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for i in idx:
        t = traj[i]
        ax.plot(t[:, 1], t[:, 0], color=S1, linewidth=0.6, alpha=0.18)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_color(AXIS)
    finish(ax, "400 Copenhagen trajectories, overlaid",
           "Each line is one run, drawn at 18% opacity")
    save(fig, "b8_overlaid_routes.png")


# ====================================================== c — benchmark

CPH_KEY = "Copenhagen city label / 1000 m cells"
MODELS = ["Random", "ItemKNN", "SVD", "ALS", "BPR", "MostPopular"]


def c1_benchmark(bench):
    entry = bench["markets"][CPH_KEY]
    vals = [entry["results"][m]["HR@10"] * 100 for m in MODELS]
    ceiling = entry["ceiling_HR"] * 100
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = [S2 if m in ("Random", "MostPopular") else S1 for m in MODELS]
    bars = ax.barh(MODELS, vals, color=colors, height=0.66)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", color=INK, fontsize=11)
    ax.axvline(ceiling, color=MUTED, linewidth=1.4, zorder=1)
    ax.text(ceiling - 1.5, len(MODELS) - 0.35,
            f"ceiling {ceiling:.0f}%\ntest items present in training", ha="right", va="top",
            color=INK_2, fontsize=10.5, linespacing=1.5)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=S2), plt.Rectangle((0, 0), 1, 1, color=S1)]
    ax.legend(handles, ["Baseline", "Collaborative filtering"], loc="lower right",
              fontsize=10.5, labelcolor=INK_2)
    finish(ax, "HR@10 by model, Copenhagen",
           f"{entry['users']} users, {entry['routes']} routes, "
           f"{entry['test_cases']} test cases", xlabel="Hit rate at 10")
    save(fig, "c1_benchmark_copenhagen.png")


def c2_hr_vs_ndcg(bench):
    entry = bench["markets"][CPH_KEY]
    hr = [entry["results"][m]["HR@10"] * 100 for m in MODELS]
    nd = [entry["results"][m]["NDCG@10"] * 100 for m in MODELS]
    y = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(y + 0.19, hr, height=0.34, color=S1, label="HR@10")
    ax.barh(y - 0.19, nd, height=0.34, color=S2, label="NDCG@10")
    for i, (h, n) in enumerate(zip(hr, nd)):
        ax.text(h + 1, i + 0.19, f"{h:.1f}", va="center", color=INK_2, fontsize=9.5)
        ax.text(n + 1, i - 0.19, f"{n:.1f}", va="center", color=INK_2, fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels(MODELS)
    ax.set_xlim(0, 50); ax.grid(axis="y", visible=False)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    ax.legend(loc="lower right", fontsize=10.5, labelcolor=INK_2)
    finish(ax, "HR@10 and NDCG@10 by model",
           "Copenhagen · NDCG discounts by rank within the top ten",
           xlabel="Score")
    save(fig, "c2_hr_vs_ndcg.png")


def c3_all_markets(bench):
    markets = list(bench["markets"])
    short = [m.replace("Copenhagen city label", "city label")
              .replace("Copenhagen ", "").replace(" cells", "") for m in markets]
    mat = np.array([[bench["markets"][mk]["results"][mo]["HR@10"] * 100
                     for mk in markets] for mo in MODELS])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    im = ax.imshow(mat, cmap=BLUES, aspect="auto", vmin=0, vmax=mat.max())
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if v > mat.max() * 0.55 else INK)
    ax.set_xticks(range(len(short))); ax.set_xticklabels(short, fontsize=9.5)
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS)
    ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED)
    finish(ax, "HR@10 by model and configuration",
           "Every market and grid resolution tested")
    save(fig, "c3_all_markets.png")


def c4_ceiling(bench):
    markets = list(bench["markets"])
    short = [m.replace("Copenhagen city label", "city label")
              .replace("Copenhagen ", "").replace(" cells", "") for m in markets]
    ceil = [bench["markets"][m]["ceiling_HR"] * 100 for m in markets]
    best = [max(bench["markets"][m]["results"][mo]["HR@10"] for mo in MODELS) * 100
            for m in markets]
    y = np.arange(len(markets))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y, ceil, height=0.6, color=DIM, label="Reachable at all (ceiling)")
    ax.barh(y, best, height=0.6, color=S1, label="Best model achieved")
    for i, (c, b) in enumerate(zip(ceil, best)):
        ax.text(c + 1, i, f"{c:.0f}%", va="center", color=INK_2, fontsize=10)
        ax.text(b - 1.5, i, f"{b:.0f}%", va="center", ha="right", color="#ffffff",
                fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(short, fontsize=9.5)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=10.5, labelcolor=INK_2)
    finish(ax, "Ceiling and best achieved HR@10",
           "Grey: test items present in training. Blue: best model result.",
           xlabel="Share of test cases")
    save(fig, "c4_ceiling_vs_achieved.png")


def main():
    style()
    res = os.path.join(ROOT, "results")
    sharing = json.load(open(os.path.join(res, "sharing_analysis.json"), encoding="utf-8"))
    bench = json.load(open(os.path.join(res, "cf_benchmark.json"), encoding="utf-8"))
    df = pd.read_csv(os.path.join(ROOT, "data", "workouts.csv"))

    print("rendering figures ...")
    a1_countries_gender(df)
    a2_distance_duration(df)
    a3_runs_per_user(df)
    a4_temporal(df)
    a5_example_trajectory()
    a6_start_points(df)

    b1_coverage_vs_grid(sharing)
    b2_users_per_route(df)
    b3_coverage_by_key(sharing)
    b4_city_dispersion(df)
    b5_top_cities(df)
    b6_users_vs_coverage(sharing)
    b7_radius_effect(sharing)
    b8_overlaid_routes(df)

    c1_benchmark(bench)
    c2_hr_vs_ndcg(bench)
    c3_all_markets(bench)
    c4_ceiling(bench)
    print("done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assign clusters to full merged CSV without pandas/sklearn.

Uses `analysis/advanced_outputs/kmeans_centers.csv` as cluster centers
and streams `analysis/merged_full.csv` with the standard library CSV reader.
This avoids dependencies on pandas/scikit-learn and writes the same outputs
as the other script: CSV of counts and a LaTeX fragment.
"""
import csv
import os
import sys
from collections import Counter, defaultdict
import math

ROOT = os.path.dirname(os.path.dirname(__file__))
ADV_OUT = os.path.join(ROOT, "analysis", "advanced_outputs")
MERGED_FULL = os.path.join(ROOT, "analysis", "merged_full.csv")
CENTERS_CSV = os.path.join(ADV_OUT, "kmeans_centers.csv")
OUT_CSV = os.path.join(ROOT, "analysis", "advanced_outputs", "top_agents_per_cluster_full.csv")
OUT_TEX = os.path.join(ROOT, "paper", "cluster_agents_table.tex")

FEATURES = [
    "average_combat_score",
    "average_damage_per_round",
    "kills_per_round",
    "headshot_percentage",
    "assists_per_round",
    "rounds_played",
]


def load_centers():
    centers = {}
    with open(CENTERS_CSV, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            cid = int(row['cluster'])
            centers[cid] = {f: float(row.get(f, 0.0)) for f in FEATURES}
    if not centers:
        raise FileNotFoundError("No centers found in kmeans_centers.csv")
    # compute per-feature fallback (mean across centers)
    fallback = {}
    for f in FEATURES:
        s = 0.0
        n = 0
        for c in centers.values():
            if f in c:
                s += c[f]
                n += 1
        fallback[f] = s / n if n else 0.0
    return centers, fallback


def parse_float(x, fallback):
    if x is None:
        return fallback
    x = x.strip()
    if x == '' or x.lower() in ('na', 'nan', 'none'):
        return fallback
    try:
        return float(x)
    except Exception:
        # strip percent
        if x.endswith('%'):
            try:
                return float(x[:-1]) * 0.01
            except Exception:
                return fallback
        return fallback


def assign_and_count(centers, fallback):
    counters = defaultdict(Counter)
    if not os.path.exists(MERGED_FULL):
        raise FileNotFoundError("merged_full.csv not found at expected location")

    with open(MERGED_FULL, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            agent = row.get('agent', '') or ''
            agent = agent.strip().lower()
            feats = []
            for feat in FEATURES:
                v = parse_float(row.get(feat), fallback[feat])
                feats.append(v)
            # compute distances
            best = None
            bestd = None
            for cid, center in centers.items():
                d = 0.0
                for v, f in zip(feats, FEATURES):
                    dv = v - center[f]
                    d += dv * dv
                if best is None or d < bestd:
                    best = cid
                    bestd = d
            counters[best][agent] += 1
            if i % 100000 == 0:
                print(f"Processed {i} rows...")

    return counters


def write_outputs(counters, topn=3):
    # write CSV
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cluster', 'agent', 'count'])
        writer.writeheader()
        for cluster in sorted(counters.keys()):
            for agent, cnt in counters[cluster].most_common():
                writer.writerow({'cluster': cluster, 'agent': agent, 'count': cnt})

    # write LaTeX
    with open(OUT_TEX, 'w') as f:
        f.write('\\begin{table}[ht]\n')
        f.write('\\centering\n')
        f.write('\\caption{Top %d agents (by snapshot count) per cluster — full merged data}\n' % topn)
        f.write('\\begin{tabular}{rl}\n')
        f.write('\\toprule\n')
        f.write('Cluster & Top agents \\\n')
        f.write('\\midrule\n')
        for cluster in sorted(counters.keys()):
            most = counters[cluster].most_common(topn)
            agents = ', '.join([a for a, _ in most])
            f.write(f"{cluster} & {agents} \\\n")
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\label{tab:cluster_agents_full}\n')
        f.write('\\end{table}\n')


def main():
    topn = 3
    if len(sys.argv) > 1:
        try:
            topn = int(sys.argv[1])
        except Exception:
            pass
    centers, fallback = load_centers()
    counters = assign_and_count(centers, fallback)
    write_outputs(counters, topn=topn)
    print('Wrote:', OUT_CSV)
    print('Wrote:', OUT_TEX)


if __name__ == '__main__':
    main()

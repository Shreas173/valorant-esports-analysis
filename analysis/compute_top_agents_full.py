#!/usr/bin/env python3
"""Compute top agents per cluster on the full merged dataset.

Loads `analysis/advanced_outputs/scaler_model.joblib` and
`analysis/advanced_outputs/kmeans_model.joblib`, then streams
`analysis/merged_full.csv` in chunks, assigns clusters, and
aggregates agent counts per cluster. Writes CSV and a LaTeX table
fragment to `paper/cluster_agents_table.tex`.
"""
import os
import sys
from collections import Counter, defaultdict

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from joblib import load
    _USE_JOBLIB = True
except Exception:
    import pickle
    _USE_JOBLIB = False


ROOT = os.path.dirname(os.path.dirname(__file__))
ADV_OUT = os.path.join(ROOT, "analysis", "advanced_outputs")
MERGED_FULL = os.path.join(ROOT, "analysis", "merged_full.csv")
MERGED_SAMPLE = os.path.join(ROOT, "analysis", "merged_sample.csv")
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


def load_models():
    scaler_path = os.path.join(ADV_OUT, "scaler_model.joblib")
    kmeans_path = os.path.join(ADV_OUT, "kmeans_model.joblib")
    if not os.path.exists(scaler_path) or not os.path.exists(kmeans_path):
        raise FileNotFoundError("Missing scaler or kmeans model in analysis/advanced_outputs")
    if _USE_JOBLIB:
        scaler = load(scaler_path)
        kmeans = load(kmeans_path)
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(kmeans_path, 'rb') as f:
            kmeans = pickle.load(f)
    return scaler, kmeans


def get_expected_features(scaler):
    # sklearn scalers have `feature_names_in_` after fit; otherwise fall back to default FEATURES
    if hasattr(scaler, 'feature_names_in_'):
        return [str(x) for x in scaler.feature_names_in_]
    return FEATURES


def compute_medians_from_sample():
    # Fallback to merged_sample to compute reasonable medians for imputation
    if not os.path.exists(MERGED_SAMPLE):
        return {f: 0.0 for f in FEATURES}
    # attempt to infer expected features from the saved scaler if possible
    # load scaler to get feature names
    scaler_path = os.path.join(ADV_OUT, "scaler_model.joblib")
    if os.path.exists(scaler_path):
        try:
            s = load(scaler_path) if _USE_JOBLIB else pickle.load(open(scaler_path, 'rb'))
            expected = get_expected_features(s)
        except Exception:
            expected = FEATURES
    else:
        expected = FEATURES

    df0 = pd.read_csv(MERGED_SAMPLE, nrows=0)
    cols = [c for c in expected if c in df0.columns]
    if cols:
        df = pd.read_csv(MERGED_SAMPLE, usecols=cols)
    else:
        df = pd.DataFrame()
    med = {}
    for f in expected:
        if f in df.columns:
            med[f] = float(df[f].median(skipna=True))
        else:
            med[f] = 0.0
    return med


def stream_and_count(scaler, kmeans, medians, chunksize=100000):
    counters = defaultdict(Counter)
    if pd is None:
        raise RuntimeError("This script requires pandas to stream the full CSV efficiently.")

    expected = get_expected_features(scaler)
    usecols = expected + ["agent"]
    # read in chunks
    for chunk in pd.read_csv(MERGED_FULL, usecols=lambda c: c in usecols, chunksize=chunksize):
        # normalize agent
        if "agent" not in chunk.columns:
            raise KeyError("agent column not found in merged_full.csv")
        chunk["agent"] = chunk["agent"].astype(str).str.strip().str.lower()
        # ensure all expected feature columns exist
        for f in expected:
            if f not in chunk.columns:
                chunk[f] = medians.get(f, 0.0)
        X = chunk[expected].astype(float).fillna(value=medians)
        Xs = scaler.transform(X)
        labels = kmeans.predict(Xs)
        for lab, ag in zip(labels, chunk["agent"]):
            counters[int(lab)][ag] += 1

    return counters


def write_outputs(counters, topn=3):
    rows = []
    for cluster in sorted(counters.keys()):
        most = counters[cluster].most_common(topn)
        agents = ", ".join([a for a, _ in most])
        rows.append({"cluster": cluster, "top_agents": agents})

    # write CSV
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster", "agent", "count"])
        writer.writeheader()
        for cluster, counter in sorted(counters.items()):
            for agent, cnt in counter.most_common():
                writer.writerow({"cluster": cluster, "agent": agent, "count": cnt})

    # write LaTeX fragment (simple table)
    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Top %d agents (by snapshot count) per cluster — full merged data}\n" % topn)
        f.write("\\begin{tabular}{rl}\n")
        f.write("\\toprule\n")
        f.write("Cluster & Top agents \\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(f"{r['cluster']} & {r['top_agents']} \\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:cluster_agents_full}\n")
        f.write("\\end{table}\n")


def main():
    topn = 3
    if len(sys.argv) > 1:
        try:
            topn = int(sys.argv[1])
        except ValueError:
            pass

    scaler, kmeans = load_models()
    medians = compute_medians_from_sample()
    counters = stream_and_count(scaler, kmeans, medians)
    write_outputs(counters, topn=topn)
    print("Wrote:", OUT_CSV)
    print("Wrote:", OUT_TEX)


if __name__ == '__main__':
    main()

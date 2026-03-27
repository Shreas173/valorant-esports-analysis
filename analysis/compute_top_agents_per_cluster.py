#!/usr/bin/env python3
"""Compute top-N agents per KMeans cluster and write CSV + LaTeX fragment.

Outputs:
- analysis/advanced_outputs/top_agents_per_cluster.csv
- paper/cluster_agents_table.tex
"""
from pathlib import Path
try:
    import pandas as pd
except Exception:
    pd = None
    import csv
    from collections import Counter, defaultdict

ROOT = Path('.')
MERGED = ROOT / 'analysis' / 'merged_sample.csv'
CLUST = ROOT / 'analysis' / 'advanced_outputs' / 'clustered_sample.csv'
OUT_CSV = ROOT / 'analysis' / 'advanced_outputs' / 'top_agents_per_cluster.csv'
OUT_TEX = ROOT / 'paper' / 'cluster_agents_table.tex'

def main(n=10):
    if pd is not None:
        merged = pd.read_csv(MERGED)
        clusters = pd.read_csv(CLUST)

        if len(merged) != len(clusters):
            print(f"Warning: length mismatch merged={len(merged)} clusters={len(clusters)}; aligning by min length")
        m = min(len(merged), len(clusters))
        merged = merged.iloc[:m].reset_index(drop=True)
        clusters = clusters.iloc[:m].reset_index(drop=True)

        merged['cluster'] = clusters['cluster']

        grp = (merged.groupby(['cluster','agent'])
                   .size()
                   .reset_index(name='count')
                   .sort_values(['cluster','count'], ascending=[True, False]))

        top_n = grp.groupby('cluster').head(n).reset_index(drop=True)
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        top_n.to_csv(OUT_CSV, index=False)
        print('Wrote', OUT_CSV)

        agent_lists = (grp.groupby('cluster')
                  .apply(lambda df: ', '.join(df['agent'].head(n)))
                  .reset_index(name='top_agents'))
    else:
        # Fallback without pandas: use csv reader and Counters
        with open(MERGED, newline='') as f1, open(CLUST, newline='') as f2:
            r1 = csv.DictReader(f1)
            r2 = csv.DictReader(f2)
            counts = defaultdict(Counter)
            i = 0
            for row1, row2 in zip(r1, r2):
                agent = row1.get('agent') or row1.get('agents') or ''
                cluster = row2.get('cluster')
                if cluster is None:
                    continue
                try:
                    cl = int(float(cluster))
                except Exception:
                    cl = cluster
                counts[cl][agent] += 1
                i += 1

        rows = []
        for cl in sorted(counts.keys()):
            for agent, cnt in counts[cl].most_common(n):
                rows.append({'cluster':cl, 'agent':agent, 'count':cnt})
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, 'w', newline='') as outf:
            writer = csv.DictWriter(outf, fieldnames=['cluster','agent','count'])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print('Wrote', OUT_CSV)

        # prepare agent_lists for LaTeX
        agent_lists = []
        for cl in sorted(counts.keys()):
            agents = ', '.join([a for a, _ in counts[cl].most_common(n)])
            agent_lists.append({'cluster':cl, 'top_agents':agents})

    # Normalize to list of records for LaTeX rendering
    if pd is not None:
        agent_records = agent_lists.to_dict(orient='records')
    else:
        agent_records = agent_lists

    # Build LaTeX table
    lines = []
    lines.append('\\begin{table}[ht]')
    lines.append('\\centering')
    lines.append('\\caption{Top %d agents (by snapshot count) per cluster}' % n)
    lines.append('\\begin{tabular}{rl}')
    lines.append('\\toprule')
    lines.append('Cluster & Top agents \\\\')
    lines.append('\\midrule')
    for row in agent_records:
        cl = int(row['cluster'])
        agents = row['top_agents'].replace('_','\\_')
        lines.append(f'{cl} & {agents} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\label{tab:cluster_agents}')
    lines.append('\\end{table}')

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', OUT_TEX)
    if pd is not None:
        try:
            print(agent_lists.to_string(index=False))
        except Exception:
            print(agent_lists)
    else:
        import json
        print(json.dumps(agent_records, indent=2))

if __name__ == '__main__':
    import sys
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    except Exception:
        n = 10
    main(n)

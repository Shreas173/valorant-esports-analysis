#!/usr/bin/env python3
from collections import Counter
import csv
import os

ROOT = os.path.dirname(__file__)
MERGED_FULL = os.path.join(ROOT, 'merged_full.csv')
OUT_AGENTS = os.path.join(ROOT, 'top_agents_overall.csv')
OUT_ORGS = os.path.join(ROOT, 'top_orgs_overall.csv')

def main():
    with open(MERGED_FULL, newline='') as f:
        r = csv.reader(f)
        hdr = next(r)
        agent_idx = None
        org_idx = None
        for i,c in enumerate(hdr):
            if c.strip().lower() == 'agent': agent_idx = i
            if c.strip().lower() == 'org': org_idx = i

        ag = Counter(); org = Counter()
        for row in r:
            if agent_idx is not None and agent_idx < len(row):
                ag[row[agent_idx].strip().lower()] += 1
            if org_idx is not None and org_idx < len(row):
                org[row[org_idx].strip()] += 1

    with open(OUT_AGENTS, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['agent','count'])
        for a,c in ag.most_common():
            w.writerow([a,c])

    with open(OUT_ORGS, 'w', newline='') as f:
        w2 = csv.writer(f)
        w2.writerow(['org','count'])
        for o,c in org.most_common():
            w2.writerow([o,c])

    print('Top 10 agents:')
    for a,c in ag.most_common(10):
        print(a,c)
    print('\nTop 10 orgs:')
    for o,c in org.most_common(10):
        print(o,c)

if __name__ == '__main__':
    main()

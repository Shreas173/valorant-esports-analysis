#!/usr/bin/env python3
"""
Exploratory Data Analysis for Valorant merged dataset.
Generates summary stats, missingness report, top counts, correlation matrix, and figures.
Usage:
  analysis/venv/bin/python3 analysis/eda.py --input analysis/merged_sample.csv --out_dir analysis/figures --report analysis/eda_report.txt
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out_dir', default='figures')
    parser.add_argument('--report', default='eda_report.txt')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    ensure_dir(args.out_dir)

    report_lines = []

    report_lines.append(f'Input rows: {df.shape[0]} columns: {df.shape[1]}')
    report_lines.append('\n-- Columns --')
    report_lines.append('\n'.join(df.columns.tolist()))

    # Basic dtypes and null counts
    dtypes = df.dtypes.astype(str)
    nulls = df.isna().sum()
    dtype_null = pd.DataFrame({'dtype': dtypes, 'nulls': nulls, 'null_percent': (nulls/len(df)).round(4)})
    dtype_null.to_csv('column_summary.csv')
    report_lines.append('\nSaved column_summary.csv')

    # Numeric summary
    num = df.select_dtypes(include=[np.number])
    num_desc = num.describe().T
    num_desc.to_csv('numeric_summary.csv')
    report_lines.append('\nSaved numeric_summary.csv')

    # Missingness heatmap (by column)
    plt.figure(figsize=(10,6))
    missing_pct = (df.isna().mean()*100).sort_values(ascending=False)
    missing_pct.plot.bar()
    plt.ylabel('Percent missing')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'missing_percent_by_column.png'), dpi=150)
    plt.close()
    report_lines.append('\nSaved missing_percent_by_column.png')

    # Top categories: agents and orgs
    if 'agents' in df.columns:
        top_agents = df['agents'].value_counts().head(20)
        top_agents.to_csv('top_agents.csv')
        plt.figure(figsize=(8,6))
        sns.barplot(x=top_agents.values, y=top_agents.index)
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'top_agents.png'), dpi=150)
        plt.close()
        report_lines.append('\nSaved top_agents.csv and top_agents.png')

    if 'org' in df.columns:
        top_orgs = df['org'].value_counts().head(20)
        top_orgs.to_csv('top_orgs.csv')
        plt.figure(figsize=(8,6))
        sns.barplot(x=top_orgs.values, y=top_orgs.index)
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'top_orgs.png'), dpi=150)
        plt.close()
        report_lines.append('\nSaved top_orgs.csv and top_orgs.png')

    # Choose key numeric columns for distributions
    candidate_cols = ['rating','average_combat_score','average_damage_per_round','kills_per_round','headshot_percentage','rounds_played']
    present = [c for c in candidate_cols if c in df.columns]
    for c in present:
        plt.figure(figsize=(6,4))
        sns.histplot(df[c].dropna(), kde=True, bins=40)
        plt.title(c)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'dist_{c}.png'), dpi=150)
        plt.close()
    report_lines.append('\nSaved distribution plots for: ' + ','.join(present))

    # Correlation heatmap for numeric columns (top 25 by non-null count)
    num_cols = num.columns.tolist()
    num_cols_sorted = sorted(num_cols, key=lambda c: df[c].notna().sum(), reverse=True)[:25]
    corr = df[num_cols_sorted].corr()
    corr.to_csv('correlation_matrix.csv')
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='vlag', center=0, annot=False)
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'correlation_matrix.png'), dpi=150)
    plt.close()
    report_lines.append('\nSaved correlation_matrix.csv and correlation_matrix.png')

    # Scatter pairplot for a few key features (sampled to 1000 rows)
    pair_features = [c for c in ['rating','average_combat_score','average_damage_per_round','kills_per_round','headshot_percentage'] if c in df.columns]
    if len(pair_features) >= 2:
        pair_df = df[pair_features].dropna()
        if len(pair_df) > 0:
            sample_n = min(1000, len(pair_df))
            sample = pair_df.sample(n=sample_n, random_state=0)
            try:
                sns.pairplot(sample)
                plt.savefig(os.path.join(args.out_dir, 'pairplot.png'), dpi=150)
                plt.close()
                report_lines.append('\nSaved pairplot.png')
            except Exception:
                report_lines.append('\nPairplot failed (likely too many unique values); skipping')
        else:
            report_lines.append('\nNot enough non-null rows for pairplot; skipping')

    # Save small overview CSVs
    df.head(100).to_csv('sample_head100.csv', index=False)
    report_lines.append('\nSaved sample_head100.csv')

    # Write textual report
    with open(args.report, 'w') as f:
        f.write('\n'.join(report_lines))

    print('EDA complete. Outputs in analysis/ and', args.out_dir)

if __name__ == '__main__':
    main()
